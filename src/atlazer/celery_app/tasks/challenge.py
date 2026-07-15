from __future__ import annotations

import structlog
from typing import Dict, Any, List
from celery import signature

from atlazer.celery_app.main import app, db_pool
from atlazer.models.challenge import ChunkAnswerMetadata
from atlazer.utils.stanza_chunker import chunk_answer as stanza_chunk_answer
from atlazer.config.settings import settings

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 7 — chunk_answer
# ─────────────────────────────────────────────────────────────────────────────

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

    chunks = stanza_chunk_answer(
        text=content,
        lang=None,
        semantic=True,
        download_models=False,
        embed_model_name=settings.local_embedding_model,
        min_words=15,
    )

    metadata.chunks = [{"text": chunk} for chunk in chunks]

    log.info(
        "challenge.chunk_answer.done",
        chunk_count=len(metadata.chunks),
        metadata=metadata.model_dump()
    )

    metadata_dump = metadata.model_dump()

    # Chain: chunk_answer → embed_answer
    signature(
        "atlazer.celery_app.tasks.challenge.embed_answer",
        kwargs=({"metadata": metadata_dump}),
        queue="challenge",
        immutable=False,
    ).apply_async()

    return metadata_dump


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 7 — embed_answer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.embed_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embed_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates embedding vectors for every chunk in `metadata["chunks"]`.

    Chunks are batched to stay within API token limits.
    Each chunk is enriched with an `embedding` key (list[float]).

    Returns the full metadata dict with embedded chunks.
    """
    chunks = metadata.get("chunks", [])
    embedded_chunks: List[Dict[str, Any]] = []

    if not chunks:
        log.warning("challenge.embed_answer.no_chunks", metadata=metadata)
        return metadata

    log.info(
        "challenge.embed_answer.start",
        chunks=len(chunks),
        provider=settings.embedding_provider,
        metadata=metadata,
    )

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "challenge.embed_answer.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "challenge.embed_answer.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks

    return metadata
