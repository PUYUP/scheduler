from __future__ import annotations

import structlog

from typing import List, Dict, Any
from atlazer.celery_app.main import app

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
    log.info("matcher.paper_for_user.start", profile_id=profile_id)
    
    pass


def _get_profiles() -> List[Dict[str, Any]]:
    """Get profiles that are ready for matching"""
    pass
