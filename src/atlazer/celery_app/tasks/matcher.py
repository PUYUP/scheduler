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
def single_user(self, profile_id: str) -> List[Dict[str, Any]]:
    log.info("matcher.single_user.start", profile_id=profile_id)

    try:
        depot = UserDepot(db_pool)
        profile = depot.get_profile(profile_id)

        if not profile:
            log.error("matcher.single_user.no_profile", profile_id=profile_id)
            return {}

        embed = profile.interest_embedding
        if embed is None:
            log.error("matcher.single_user.no_embedding", profile_id=profile_id)
            return {}

        matcher_depot = MatcherDepot(db_pool)
        results = matcher_depot.match_papers_by_interest(embed)

        for r in results:
            log.info(
                "matcher.single_user.match",
                profile_id=profile_id,
                pdf_url=r.pdf_url,
                title=r.title,
            )

        log.info(
            "matcher.single_user.success",
            profile_id=profile_id,
            matches_count=len(results),
        )

        return [
            {
                "id": r.id,
                "pdf_url": r.pdf_url,
                "title": r.title,
            }
            for r in results
        ]

    except Exception as e:
        log.error("matcher.single_user.failed", profile_id=profile_id, error=str(e))
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
            return {}

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

            for r in results:
                log.info(
                    "matcher.batch_user.match",
                    profile_id=profile_id,
                    pdf_url=r.pdf_url,
                    title=r.title,
                )

            processed_count += 1

        log.info(
            "matcher.batch_user.success",
            processed_count=processed_count,
            skipped_count=skipped_count,
        )
        return {}

    except Exception as e:
        log.error("matcher.batch_user.failed", error=str(e))
        raise self.retry(exc=e)


def _get_profiles(self) -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching"""
    depot = UserDepot(db_pool)
    return depot.get_profiles_for_paper_matching()