from __future__ import annotations

import structlog

from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import select, or_

from atlazer.celery_app.main import app
from atlazer.models.user import ProfileORM
from atlazer.celery_app.main import db_pool
from atlazer.storage.matcher import MatcherDepot
from atlazer.storage.user import UserDepot

log = structlog.get_logger(__name__)


def _serialize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ubah list match (dict berisi paper + distance + relevance_score) menjadi
    list dict yang siap dikirim/di-log.
    """
    serialized = []
    for m in matches:
        paper = m["paper"]
        serialized.append(
            {
                "id": paper.id,
                "pdf_url": paper.pdf_url,
                "title": paper.title,
                "distance": m["distance"],
                "relevance_score": m["relevance_score"],
            }
        )
    return serialized


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
def single_user(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    log.info("matcher.single_user.start", user_id=user_id)

    empty_result: Dict[str, List[Dict[str, Any]]] = {"closest": [], "farthest": []}

    try:
        depot = UserDepot(db_pool)
        profile = depot.get_profile_by_user_id(user_id)

        if not profile:
            log.error("matcher.single_user.no_profile", user_id=user_id)
            return empty_result

        embed = profile.interest_embedding
        if embed is None:
            log.error("matcher.single_user.no_embedding", user_id=user_id)
            return empty_result

        matcher_depot = MatcherDepot(db_pool)
        results = matcher_depot.match_papers_by_interest(embed)

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

        matches_count = len(results["closest"]) + len(results["farthest"])

        log.info(
            "matcher.single_user.success",
            user_id=user_id,
            matches_count=matches_count,
        )

        return {
            "closest": _serialize_matches(results["closest"]),
            "farthest": _serialize_matches(results["farthest"]),
        }

    except Exception as e:
        log.error("matcher.single_user.failed", user_id=user_id, error=str(e))
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
def batch_user(self) -> Dict[str, Any]:
    log.info("matcher.batch_user.start")

    try:
        profiles = _get_profiles(self)

        if not profiles:
            log.info("matcher.batch_user.no_profiles")
            return {"processed_count": 0, "skipped_count": 0}

        matcher_depot = MatcherDepot(db_pool)
        processed_count = 0
        skipped_count = 0

        for prof in profiles:
            profile_id = prof.get("id")
            embed = prof.get("interest_embedding")

            if embed is None:
                log.error(
                    "matcher.batch_user.no_embedding",
                    profile_id=profile_id,
                )
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
        return {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
        }

    except Exception as e:
        log.error("matcher.batch_user.failed", error=str(e))
        raise self.retry(exc=e)


def _get_profiles(self) -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching"""
    depot = UserDepot(db_pool)
    return depot.get_profiles_for_paper_matching()