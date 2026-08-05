from __future__ import annotations
from typing import Any, Dict

import structlog

from celery import group, signature
from atlazer.celery_app.main import app
from atlazer.ingestion.providers.iaescore import IaeScoreProvider

log = structlog.get_logger(__name__)
provider = IaeScoreProvider()
repository = provider.provider_name


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 5 — fetch_page
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.iaescore.fetch_page",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="iaescore",
    ignore_result=False,
)
def fetch_page(self) -> Dict[str, Any]:
    result = provider.fetch_page()
    return {}
