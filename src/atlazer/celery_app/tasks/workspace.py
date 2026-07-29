
from __future__ import annotations

import uuid
import structlog
from typing import Dict, Any, List
from celery import group, signature

from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content as stanza_chunk_context
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.models.workspace import (
    ChunkContextMetadata,
    ContextChunkORM
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
    validated = ChunkContextMetadata.model_validate(metadata)
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


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 7 — embed_context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.embed_context",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embed_context(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = metadata.get("chunks", [])

    if not chunks:
        raise ValueError("No chunks to embed")

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "workspace.embed_context.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "workspace.embed_context.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 7 — save embedding context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.save_embedding_context",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding_context(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.save_embedding_context.start")

    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    context_id = metadata.get('context_id')
    workspace_id = metadata.get('workspace_id')

    if not chunks:
        log.warning("workspace.save_embedding_context.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not context_id or not workspace_id:
        log.warning("workspace.save_embedding_context.missing_user_id_or_context_id_or_workspace_id", metadata=metadata)
        raise ValueError("Missing user_id or context_id or workspace_id")
    
    try:
        user_uuid = uuid.UUID(str(user_id))
        context_uuid = uuid.UUID(str(context_id))
        workspace_uuid = uuid.UUID(str(workspace_id))
    except ValueError as exc:
        log.error("workspace.save_embedding_context.invalid_uuid", metadata=metadata, error=str(exc))
        raise ValueError("Invalid UUID string format")

    log.info("workspace.save_embedding_context.mapping_payloads")
    payloads: List[ContextChunkORM] = []
    for chunk in chunks:
        attributes = {
            "embedding_model": chunk.get("embedding_model"),
            "embedding_adapter": chunk.get("embedding_adapter"),
            "embedding_normalized": chunk.get("embedding_normalized", True),
            "token_count": chunk.get("token_count"),
            "word_count": chunk.get("word_count"),
        }

        payloads.append(ContextChunkORM(
            user_id=user_uuid,
            context_id=context_uuid,
            workspace_id=workspace_uuid,
            content=chunk.get('text'),
            embedding=chunk.get('embedding'),
            chunk_index=chunk.get('chunk_index'),
            attributes=attributes,
        ))
    
    log.info("workspace.save_embedding_context.done", metadata=metadata)
    return metadata
