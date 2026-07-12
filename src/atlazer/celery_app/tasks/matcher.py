from __future__ import annotations

import structlog
from typing import List, Dict, Any

from celery import group
from atlazer.celery_app.main import app, db_pool
from atlazer.models.user import ProfileORM
from atlazer.storage.matcher import MatcherDepot
from atlazer.storage.user import UserDepot
from atlazer.storage.paper import PaperDepot
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

    log.info("matcher.single_user.start", user_id=user_id, language_code=language_code)

    empty_result: Dict[str, List[Dict[str, Any]]] = {"closest": [], "farthest": []}

    if not user_id:
        log.error("matcher.single_user.missing_user_id")
        return {**metadata, **empty_result}

    try:
        profile = UserDepot(db_pool).get_profile_by_user_id(user_id)
        
        # Guard clause: check profile and embedding simultaneously
        if profile is None or profile.interest_embedding is None:
            log.error(
                "matcher.single_user.no_profile_or_embedding", 
                user_id=user_id, 
                has_profile=bool(profile)
            )
            return {**metadata, **empty_result}

        results = MatcherDepot(db_pool).match_papers_by_interest(profile.interest_embedding)
        tasks = []

        for label, matches in results.items():
            for m in matches:
                paper = m["paper"]
                log.info(
                    "matcher.single_user.match",
                    user_id=user_id,
                    category=label,
                    pdf_url=paper.pdf_url,
                    title=paper.title,
                    relevance_score=m["relevance_score"],
                )

                tasks.append(
                    summarize_paper.s(
                        metadata={
                            "user_id": user_id,
                            "paper_id": str(paper.id),
                            "language_code": language_code,
                        },
                    ).set(queue="matcher")
                )

        if tasks:
            group(tasks).apply_async()
    
        # Use .get() defensively in case the depot returns varying keys
        matches_count = len(results.get("closest", [])) + len(results.get("farthest", []))

        log.info("matcher.single_user.success", user_id=user_id, matches_count=matches_count)

        metadata.update({
            "closest": _serialize_matches(results.get("closest", [])),
            "farthest": _serialize_matches(results.get("farthest", [])),
        })
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

    try:
        profiles = _get_profiles()
        if not profiles:
            log.info("matcher.batch_user.no_profiles")
            return {"processed_count": 0, "skipped_count": 0}

        matcher_depot = MatcherDepot(db_pool)
        processed_count = 0
        skipped_count = 0

        for prof in profiles:
            profile_id = prof.get("id")
            embed = prof.get("interest_embedding")

            if not embed:
                log.error("matcher.batch_user.no_embedding", profile_id=profile_id)
                skipped_count += 1
                continue

            results = matcher_depot.match_papers_by_interest(embed)

            for label, matches in results.items():
                for m in matches:
                    paper = m["paper"]
                    log.info(
                        "matcher.batch_user.match",
                        profile_id=profile_id,
                        category=label,
                        pdf_url=paper.pdf_url,
                        title=paper.title,
                        relevance_score=m["relevance_score"],
                    )

            processed_count += 1

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
    language_code = metadata.get("language_code", "en")
    
    log.info(
        "matcher.summarize_paper.start",
        user_id=user_id,
        paper_id=paper_id,
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
            language_code=language_code
        )
        
        metadata.update({
            "chunks_count": len(chunk_contents),
            "gemini_job_id": job.name
        })
        
        return metadata
        
    except Exception as e:
        log.error("matcher.summarize_paper.failed", user_id=user_id, paper_id=paper_id, error=str(e), exc_info=True)
        raise self.retry(exc=e)