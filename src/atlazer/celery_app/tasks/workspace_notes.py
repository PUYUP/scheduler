from __future__ import annotations

import uuid
import structlog
from typing import Dict, Any, List

from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content as stanza_chunk_note
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.workspace_notes import WorkspaceNoteDepot
from atlazer.models.workspace import (
    ChunkNoteMetadata,
    NoteChunkORM,
    NotePaperORM,
    NoteSimilarityORM,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 7 — chunk_note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.chunk_note",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunk_note(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    validated = ChunkNoteMetadata.model_validate(metadata)
    content = validated.content
    language_code = validated.language_code

    log.info("workspace.chunk_note.start", metadata=validated.model_dump())

    chunks = stanza_chunk_note(
        text=content,
        lang=language_code,
        semantic=True,
        download_models=False,
        embed_model_name=settings.local_embedding_model,
        min_words=settings.context_chunker_min_words,
        max_words=settings.context_chunker_max_words,
    )

    validated.chunks = [{"text": chunk} for chunk in chunks]

    log.info(
        "workspace.chunk_note.done",
        chunk_count=len(validated.chunks),
        metadata=validated.model_dump()
    )

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 6 — embed_note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.embed_note",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embed_note(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = metadata.get("chunks", [])

    if not chunks:
        raise ValueError("No chunks to embed")

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "workspace.embed_note.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "workspace.embed_note.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 6 — save embedding note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.save_embedding_note",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding_note(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.save_embedding_note.start")

    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    note_id = metadata.get('note_id')
    workspace_id = metadata.get('workspace_id')

    if not chunks:
        log.warning("workspace.save_embedding_note.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not note_id or not workspace_id:
        log.warning("workspace.save_embedding_note.missing_user_id_or_note_id_or_workspace_id", metadata=metadata)
        raise ValueError("Missing user_id or note_id or workspace_id")
    
    try:
        user_uuid = uuid.UUID(str(user_id))
        note_uuid = uuid.UUID(str(note_id))
        workspace_uuid = uuid.UUID(str(workspace_id))
    except ValueError as exc:
        log.error("workspace.save_embedding_note.invalid_uuid", metadata=metadata, error=str(exc))
        raise ValueError("Invalid UUID string format")

    log.info("workspace.save_embedding_note.mapping_payloads")
    payloads: List[NoteChunkORM] = []
    for chunk in chunks:
        attributes = {
            "embedding_dim": chunk.get("embedding_dim"),
            "embedding_model": chunk.get("embedding_model"),
            "embedding_adapter": chunk.get("embedding_adapter"),
            "embedding_normalized": chunk.get("embedding_normalized", True),
            "token_count": chunk.get("token_count"),
            "word_count": chunk.get("word_count"),
        }

        payloads.append(NoteChunkORM(
            user_id=user_uuid,
            note_id=note_uuid,
            workspace_id=workspace_uuid,
            content=chunk.get('text'),
            embedding=chunk.get('embedding'),
            chunk_index=chunk.get('chunk_index'),
            attributes=attributes,
        ))

    try:
        depot = WorkspaceNoteDepot(db_pool)
        depot.bulk_insert_chunks(payloads)
    except Exception as exc:
        log.error(
            "workspace.save_embedding_note.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info("workspace.save_embedding_note.done", metadata=metadata)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 6 — note paper matcher
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.match_papers_by_note",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def match_papers_by_note(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.match_papers_by_note.start", metadata=metadata)

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")

    if not note_id or not workspace_id:
        log.info("workspace.missing_required_field")
        raise ValueError("Missing note_id or workspace_id")

    try:
        depot = WorkspaceNoteDepot(db_pool)
        chunks = depot.get_chunks_by_note_id(note_id)
        matcher = depot.match_note_with_papers(chunks=chunks, candidate_pool_size=1000)

        # collect all the matched data
        papers = []
        similar_chunks = []

        for m in matcher:
            papers.extend(matcher[m]["papers"])
            similar_chunks.extend(matcher[m]["similar_chunks"])

        # Convert PaperORM objects to basic dicts
        serialized_papers = [
            {
                "id": paper.get("id"), 
                "title": paper.get("title")
            } for paper in papers
        ]

        metadata["matched_result"] = {
            "papers": serialized_papers,
            "similar_chunks": similar_chunks,
        }
    except Exception as exc:
        log.error(
            "workspace.match_papers_by_note.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 6 — save matched papers for notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.save_note_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_note_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.save_note_papers.start")

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    papers = matched_result.get("papers", [])

    if papers:
        log.info("workspace.save_note_papers.inserting_data", payload_count=len(papers))
        payloads: List[NotePaperORM] = []

        # payloads enrichment
        for paper in papers:
            payload = NotePaperORM(
                paper_id=paper.get("id"),
                note_id=note_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            payloads.append(payload)

        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.bulk_insert_papers(payloads)
        except Exception as exc:
            log.error(
                "workspace.save_note_papers.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 6 — save note similarities
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.save_note_similarities",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_note_similarities(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.save_note_similarities.start")

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info("workspace.save_note_similarities.inserting_data", payload_count=len(similar_chunks))
        payloads: List[NoteSimilarityORM] = []

        # payloads enrichment
        for similarity in similar_chunks:
            payload = NoteSimilarityORM(
                workspace_id=workspace_id,
                user_id=user_id,
                paper_id=similarity.get("paper_id"),
                note_id=note_id,
                note_chunk_id=similarity.get("chunk_id"),
                note_content=similarity.get("chunk_content"),
                document_chunk_id=similarity.get("document_id"),
                document_content=similarity.get("document_content"),
                similarity_score=similarity.get("similarity_score"),
                attributes=similarity.get("attributes")
            )
            payloads.append(payload)

        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.bulk_insert_similarities(payloads)
        except Exception as exc:
            log.error(
                "workspace.save_note_similarities.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata
