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
    name="atlazer.celery_app.tasks.matcher.paper_for_user",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="matcher",
    ignore_result=False,
)
def paper_for_user(self, profile_id: str) -> Dict[str, Any]:
    log.info("matcher.paper_for_user.start")
    
    depot = UserDepot(db_pool)
    profile = depot.get_profile(profile_id)
    embed = profile.interest_embedding

    if not profile:
        log.error("matcher.paper_for_user.no_profile", profile_id=profile_id)
        return {}
    
    if embed is None:
        log.error("matcher.paper_for_user.no_embedding", profile_id=profile_id)
        return {}

    depot = MatcherDepot(db_pool)
    results = depot.match_papers_by_interest(embed)
    for r in results:
        print(r.pdf_url, ':', r.title)
    return {}


@app.task(
    name="atlazer.celery_app.tasks.matcher.paper_for_users",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="matcher",
    ignore_result=False,
)
def paper_for_users(self) -> Dict[str, Any]:
    log.info("matcher.paper_for_users.start")
    
    profiles = _get_profiles(self)
    prof_a = profiles[0]
    print(prof_a.get("id", None))
    embed = prof_a.get("interest_embedding", None)
    
    depot = MatcherDepot(db_pool)
    results = depot.match_papers_by_interest(embed)
    for r in results:
        print(r.pdf_url, ':', r.title)
    return {}


def _get_profiles(self) -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching"""
    depot = UserDepot(db_pool)
    return depot.get_profiles_for_paper_matching()
