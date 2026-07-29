from __future__ import annotations

import uuid
import structlog
import numpy as np
from typing import Dict, Any, List
from celery import group, signature
from sklearn.metrics.pairwise import cosine_similarity

from atlazer.celery_app.main import app, db_pool
from atlazer.celery_app.tasks.evaluation import generate_jsonl
from atlazer.utils.stanza_chunker import chunk_content as stanza_chunk_context
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.challenge import ChallengeDepot
from atlazer.models.challenge import (
    ChunkAnswerMetadata,
    AnswerChunkORM,
    AnswerSimilarityORM
)
from atlazer.utils.answer_scoring import (
    getting_answer_chunks,
    getting_paper_chunks,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 7 — chunk_context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.chunk_context",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunk_context(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    validated = ChunkAnswerMetadata.model_validate(metadata)
    content = validated.content
    language_code = validated.language_code

    log.info("workspace.chunk_context.start", metadata=validated.model_dump())

    chunks = stanza_chunk_context(
        text=content,
        lang=language_code,
        semantic=True,
        download_models=False,
        embed_model_name=settings.local_embedding_model,
        min_words=1,
        max_words=35,
    )

    validated.chunks = [{"text": chunk} for chunk in chunks]

    log.info(
        "workspace.chunk_context.done",
        chunk_count=len(validated.chunks),
        metadata=validated.model_dump()
    )

    return validated.model_dump()