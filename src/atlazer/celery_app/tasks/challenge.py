from __future__ import annotations

import uuid
import structlog
from typing import Dict, Any, List
from celery import signature

from atlazer.celery_app.main import app, db_pool
from atlazer.models.challenge import ChunkAnswerMetadata
from atlazer.utils.stanza_chunker import chunk_answer as stanza_chunk_answer
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.challenge import ChallengeDepot
from atlazer.models.challenge import AnswerChunkORM

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
        lang=language_code,
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


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 7 — save embedding answer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.save_embedding_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    challenge_id = metadata.get('challenge_id')

    if not chunks:
        log.warning("challenge.save_embedding_answer.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not challenge_id:
        log.warning("challenge.save_embedding_answer.missing_user_id_or_challenge_id", metadata=metadata)
        raise ValueError("Missing user_id or challenge_id")

    depot = ChallengeDepot(db_pool)
    saved_results = []
    
    try:
        user_uuid = uuid.UUID(str(user_id))
        challenge_uuid = uuid.UUID(str(challenge_id))
    except ValueError as exc:
        log.error("challenge.save_embedding_answer.invalid_uuid", metadata=metadata, error=str(exc))
        raise ValueError("Invalid UUID string format for user_id or challenge_id")

    for chunk in chunks:
        answer_chunk = AnswerChunkORM(
            user_id=user_uuid,
            challenge_id=challenge_uuid,
            content=chunk.get("text"),
            embedding=chunk.get("embedding"),
            embedding_model=chunk.get("embedding_model"),
            embedding_adapter=chunk.get("embedding_adapter"),
            embedding_normalized=chunk.get("embedding_normalized", True),
            token_count=chunk.get("token_count"),
            word_count=chunk.get("word_count"),
        )
        try:
            res = depot.save_embedding_answer(answer_chunk)
            saved_results.append(res)
        except Exception as exc:
            log.error(
                "challenge.save_embedding_answer.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    metadata["saved_chunks"] = saved_results
    log.info("challenge.save_embedding_answer.done", metadata=metadata)

    return metadata
