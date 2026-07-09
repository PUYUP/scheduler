from __future__ import annotations

import structlog

from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import select, or_

from atlazer.celery_app.main import app
from atlazer.models.user import ProfileORM
from atlazer.celery_app.main import db_pool
from atlazer.storage.matcher import MatcherDepot

log = structlog.get_logger(__name__)


@app.task(
    name="atlazer.celery_app.tasks.matcher.paper_for_user",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="matcher",
    ignore_result=False,
)
def paper_for_user(self) -> Dict[str, Any]:
    log.info("matcher.paper_for_user.start")
    
    profiles = _get_profiles(self)
    prof_a = profiles[1]
    print(prof_a.get("id", None))
    embed = prof_a.get("interest_embedding", None)
    
    depot = MatcherDepot(db_pool)
    results = depot.match_papers_by_interest(embed)
    for r in results:
        print(r.pdf_url, ':', r.title)
    return {}


def _get_profiles(self) -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching"""
    current_time = datetime.now(timezone.utc).isoformat()
    stmt = select(ProfileORM).where(
        or_(
            ProfileORM.next_processed_at == None,
            ProfileORM.next_processed_at < current_time
        )
    ).limit(10)

    with db_pool.session() as session:
        try:
            profiles = session.execute(stmt).scalars().all()

            return [
                {
                    "id": p.id,
                    "interest": p.interest,
                    "interest_embedding": p.interest_embedding,
                    "next_processed_at": p.next_processed_at.isoformat() if p.next_processed_at else None,
                }
                for p in profiles
            ]
        except Exception as e:
            session.rollback()
            log.error("matcher.paper_for_user.failed", error=str(e))
            raise self.retry(exc=e)
