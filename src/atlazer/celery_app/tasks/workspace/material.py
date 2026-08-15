from __future__ import annotations

import numpy as np
import uuid
import structlog
import json
import re

from datetime import datetime, date, time, timedelta
from typing import Dict, Any, List
from pathlib import Path
from decimal import Decimal
from uuid import UUID
from collections import defaultdict
from celery import group, chord

from atlazer.utils.gemini_batch import upload_chunk_file, process_jsonl_file, get_batch_results
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.workspace.notes import WorkspaceNoteDepot, NoteEnrichedChunkORM
from atlazer.storage.workspace.context import WorkspaceContextDepot
from atlazer.models.workspace import (
    ChunkNoteMetadata,
    NoteChunkORM,
    NotePaperORM,
    NoteSimilarityORM,
    NoteEnrichedORM,
    WorkspaceORM,
    NoteEnrichedSimilarityORM,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 10 — find papers
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.find_relevant_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def find_relevant_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.find_relevant_papers.start", metadata=metadata)

    workspace_id = metadata.get("workspace_id")
    enriched_note_id = metadata.get("enriched_note_id")

    if not enriched_note_id or not workspace_id:
        log.info("workspace.material.find_relevant_papers.missing_required_field")
        raise ValueError("Missing enriched_note_id or workspace_id")

    try:
        depot = WorkspaceContextDepot(db_pool)
        note_depot = WorkspaceNoteDepot(db_pool)
        chunks = note_depot.get_enriched_chunks(workspace_id=workspace_id, enriched_note_id=enriched_note_id)
        matcher = depot.match_context_with_papers(chunks=chunks, candidate_pool_size=1000)

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
            "workspace.material.find_relevant_papers.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 10 — save context similarities
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.save_similarities",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_similarities(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.save_similarities.start")

    workspace_id = metadata.get("workspace_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info(
            "workspace.context.save_similarities.inserting_data",
            payload_count=len(similar_chunks)
        )

        # payloads enrichment
        payloads: List[NoteEnrichedSimilarityORM] = []

        for similarity in similar_chunks:
            payload = NoteEnrichedSimilarityORM(
                workspace_id=workspace_id,
                enriched_note_id=similarity.get("enriched_note_id"),
                enriched_chunk_id=similarity.get("enriched_chunk_id"),
                paper_id=similarity.get("paper_id"),
                enriched_content=similarity.get("enriched_content"),
                document_chunk_id=similarity.get("document_chunk_id"),
                document_content=similarity.get("document_content"),
                similarity_score=similarity.get("similarity_score"),
                attributes=similarity.get("attributes")
            )
            payloads.append(payload)

        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.insert_enriched_similarities(payloads)
        except Exception as exc:
            log.error(
                "workspace.material.save_similarities.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata
