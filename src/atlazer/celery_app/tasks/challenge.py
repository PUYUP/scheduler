from __future__ import annotations

import structlog
from typing import Dict, Any, List

from atlazer.celery_app.main import app, db_pool
from atlazer.models.challenge import ChunkAnswerMetadata

log = structlog.get_logger()


@app.task(
    name="atlazer.celery_app.tasks.challenge.chunk_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunk_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = ChunkAnswerMetadata.model_validate(metadata)
    content = metadata.content
    language_code = metadata.language_code

    log.info("challenge.chunk_answer.start", metadata=metadata.model_dump())

    metadata.chunks = []

    return metadata.model_dump()
