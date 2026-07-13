from __future__ import annotations

import structlog
from typing import List, Dict, Any
from datetime import datetime, timedelta

from celery import group, signature
from atlazer.celery_app.main import app, db_pool
from atlazer.models.user import ProfileORM, ProfileUpdate
from atlazer.storage.matcher import MatcherDepot
from atlazer.storage.user import UserDepot
from atlazer.storage.paper import PaperDepot
from atlazer.storage.challenge import ChallengeDepot
from atlazer.utils.gemini_batch import create_batch_job

log = structlog.get_logger(__name__)


def _serialize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of matches into a serialized list of dictionaries.
    """
    return [
        {
            "id": m["paper"].id,
            "pdf_url": m["paper"].pdf_url,
            "title": m["paper"].title,
            "distance": m["distance"],
            "relevance_score": m["relevance_score"],
        }
        for m in matches
    ]


def _get_profiles() -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching."""
    return UserDepot(db_pool).get_profiles_for_paper_matching()


@app.task(
    name="atlazer.celery_app.tasks.matcher.single_user",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    soft_time_limit=60,
    time_limit=90,
    queue="matcher",
    ignore_result=False,
)
def single_user(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    user_id = metadata.get("user_id")
    language_code = metadata.get("language_code", "en")
    intereset_embedding = metadata.get("intereset_embedding")

    log.info("matcher.single_user.start", user_id=user_id, language_code=language_code)

    empty_result: Dict[str, List[Dict[str, Any]]] = {"closest": [], "farthest": []}

    if not user_id:
        log.error("matcher.single_user.missing_user_id")
        return {**metadata, **empty_result}

    try:
        # getting interest from profile
        if intereset_embedding is None:
            profile = UserDepot(db_pool).get_profile_by_user_id(user_id)
            
            # Guard clause: check profile and embedding simultaneously
            if profile is None or profile.interest_embedding is None:
                log.error(
                    "matcher.single_user.no_profile_or_embedding", 
                    user_id=user_id, 
                    has_profile=bool(profile)
                )
                return {**metadata, **empty_result}
            intereset_embedding = profile.interest_embedding

        # process with intereset embedding
        results = MatcherDepot(db_pool).match_papers_by_interest(
            user_id=user_id,
            intereset_embedding=intereset_embedding
        )

        metadata.update({
            "closest": _serialize_matches(results.get("closest", [])),
            "farthest": _serialize_matches(results.get("farthest", [])),
        })

        # create challenge
        target_date = datetime.now() + timedelta(days=2)
        challenge_depot = ChallengeDepot(db_pool)
        challenge = challenge_depot.insert_challenge(
            user_id=user_id,
            target_date=target_date,
            papers=metadata,
        )
        challenge_id = str(challenge.id)

        # map papers with challenge
        # {"paper_id": "challenge_id"}
        paper_to_challenge: Dict[str, str] = {}
        for item in challenge.challenge_papers:
            paper_id = str(item.paper_id)
            paper_to_challenge[paper_id] = str(item.id)

        metadata.update({
            "challenge_id": challenge_id,
        })

        tasks = []

        for label, matches in results.items():
            for m in matches:
                paper = m["paper"]
                paper_id = str(paper.id)
                challenge_paper_id = paper_to_challenge.get(paper_id)

                log.info(
                    "matcher.single_user.match",
                    user_id=user_id,
                    label=label,
                    paper_id=paper_id,
                    challenge_paper_id=challenge_paper_id,
                    pdf_url=paper.pdf_url,
                    title=paper.title,
                    relevance_score=m["relevance_score"],
                )

                tasks.append(
                    summarize_paper.s(
                        metadata={
                            "user_id": user_id,
                            "paper_id": paper_id,
                            "challenge_id": challenge_id,
                            "challenge_paper_id": challenge_paper_id,
                            "language_code": language_code,
                        },
                    ).set(queue="matcher")
                )

        if tasks:
            group(tasks).apply_async()
    
        # Use .get() defensively in case the depot returns varying keys
        matches_count = len(results.get("closest", [])) + len(results.get("farthest", []))

        log.info("matcher.single_user.success", user_id=user_id, matches_count=matches_count)

        return metadata

    except Exception as e:
        # Added exc_info=True for better stack traces in your logs
        log.error("matcher.single_user.failed", user_id=user_id, error=str(e), exc_info=True)
        raise self.retry(exc=e)


@app.task(
    name="atlazer.celery_app.tasks.matcher.batch_user",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    soft_time_limit=300,
    time_limit=360,
    queue="matcher",
    ignore_result=False,
)
def batch_user(self) -> Dict[str, int]:
    log.info("matcher.batch_user.start")
    user_depot = UserDepot(db_pool)

    try:
        profiles = _get_profiles()
        if not profiles:
            log.info("matcher.batch_user.no_profiles")
            return {"processed_count": 0, "skipped_count": 0}

        matcher_depot = MatcherDepot(db_pool)
        skipped_count = 0
        tasks_to_run = []
        profile_ids = []

        for prof in profiles:
            profile_id = str(prof.get("id"))
            user_id = str(prof.get("user_id"))
            embed = prof.get("interest_embedding")
            language_code = prof.get("language_code", "en")

            if embed is None or len(embed) == 0:
                log.error("matcher.batch_user.no_embedding", profile_id=profile_id)
                skipped_count += 1
                continue

            # 1. Pastikan ID menjadi string (karena JSON tidak mendukung objek UUID natively)
            # 2. Konversi NumPy array (embed) ke list python biasa
            
            payload = {
                "user_id": user_id, 
                "language_code": language_code,
            }
            
            # Jika Anda mengirimkan embedding ke dalam payload:
            if embed is not None:
                # Cek apakah itu numpy array, lalu konversi
                payload["interest_embedding"] = embed.tolist() if hasattr(embed, 'tolist') else embed

            tasks_to_run.append(single_user.s(payload).set(queue="matcher"))

            # collect profile id for bulk update
            profile_ids.append(profile_id)

        processed_count = len(tasks_to_run)

        # process user in parallel
        if tasks_to_run:
            # Bungkus semua task dalam group dan eksekusi secara paralel (asynchronous)
            job_group = group(tasks_to_run)
            result = job_group.apply_async()
            
            # PENTING: Jika Anda TIDAK butuh hasil return (metadata) di dalam task ini, 
            # hindari menggunakan `result.get()` agar tidak terjadi deadlock pada worker.
            # Biarkan task berjalan di background.
            
            # Namun, jika Anda SANGAT perlu menunggunya (misal untuk menghitung yang sukses), 
            # Anda bisa memanggil `result.get()`. (Gunakan dengan hati-hati).
            # metadata_list = result.get()

            # update profile match result
            user_depot.bulk_update_profiles(
                profile_ids,
                ProfileUpdate(
                    next_processed_at=datetime.now() + timedelta(days=2)
                )
            )

        log.info(
            "matcher.batch_user.success",
            processed_count=processed_count,
            skipped_count=skipped_count,
        )
        return {"processed_count": processed_count, "skipped_count": skipped_count}

    except Exception as e:
        log.error("matcher.batch_user.failed", error=str(e), exc_info=True)
        raise self.retry(exc=e)


@app.task(
    name="atlazer.celery_app.tasks.matcher.summarize_paper",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    soft_time_limit=300,
    time_limit=360,
    queue="matcher",
    ignore_result=False,
)
def summarize_paper(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    user_id = metadata.get("user_id")
    paper_id = metadata.get("paper_id")
    challenge_id = metadata.get("challenge_id")
    challenge_paper_id = metadata.get("challenge_paper_id")
    language_code = metadata.get("language_code", "en")
    
    log.info(
        "matcher.summarize_paper.start",
        user_id=user_id,
        paper_id=paper_id,
        challenge_id=challenge_id,
        challenge_paper_id=challenge_paper_id,
        language_code=language_code
    )
    
    try:
        chunks = PaperDepot(db_pool).get_chunks_by_paper_id(paper_id)
        if not chunks:
            log.error("matcher.summarize_paper.no_chunks", paper_id=paper_id)
            return metadata

        # Send to Gemini
        chunk_contents = [c.content for c in chunks]
        job = create_batch_job(
            documents=[chunk_contents], 
            display_name=f"paper-summary-{user_id}-{paper_id}",
            language_code=language_code,
            user_id=user_id,
            paper_id=paper_id,
            challenge_id=challenge_id,
            challenge_paper_id=challenge_paper_id,
        )
        
        metadata.update({
            "chunks_count": len(chunk_contents),
            "gemini_job_id": job.name
        })
        
        return metadata
        
    except Exception as e:
        log.error("matcher.summarize_paper.failed", user_id=user_id, paper_id=paper_id, error=str(e), exc_info=True)
        raise self.retry(exc=e)