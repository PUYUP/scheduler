from __future__ import annotations
import structlog

from typing import Dict, Any
from atlazer.celery_app.main import app

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 7 — evaluation_answers with critical thinking, etc
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.evaluate_answers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def evaluate_answers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return metadata
