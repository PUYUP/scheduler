from __future__ import annotations

import structlog

from typing import List, Dict, Any
from atlazer.celery_app.main import app

log = structlog.get_logger(__name__)


@app.task(
    name="atlazer.celery_app.tasks.matcher.paper_fitter",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="matcher",
    ignore_result=False,
)
def paper_fitter(self, profile_id: str) -> Dict[str, Any]:
    log.info("matcher.paper_fitter.start", profile_id=profile_id)
    
    pass
