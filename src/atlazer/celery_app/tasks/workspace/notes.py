from __future__ import annotations

import numpy as np
import uuid
import structlog
import json
import re

from datetime import datetime, time, timedelta
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
from celery import group, chord, signature, chain

from atlazer.utils.text_cleaner import orm_to_dict
from atlazer.utils.gemini_batch import upload_chunk_file, process_jsonl_file, get_batch_results
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.workspace.notes import WorkspaceNoteDepot, LearningMaterialChunkORM
from atlazer.storage.workspace.workspace import WorkspaceDepot
from atlazer.models.workspace import (
    ChunkNoteMetadata,
    NoteChunkORM,
    NotePaperORM,
    NoteDocumentORM,
    LearningMaterialNoteORM,
    WorkspaceORM,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 16 — chunk_note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.chunking",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunking(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    validated = ChunkNoteMetadata.model_validate(metadata)
    content = validated.content
    language_code = validated.language_code

    log.info("workspace.notes.chunking.start", metadata=validated.model_dump())

    chunks = chunk_content(
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
        "workspace.notes.chunking.done",
        chunk_count=len(validated.chunks),
        metadata=validated.model_dump()
    )

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 16 — embed_note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.embedding",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embedding(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    chunks = metadata.get("chunks", [])
    log.info("workspace.notes.embedding.start", metadata=metadata)

    if not chunks:
        raise ValueError("No chunks to embed")

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "workspace.notes.embedding.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "workspace.notes.embedding.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 16 — save embedding note
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_embedding",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_embedding.start")

    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    note_id = metadata.get('note_id')
    workspace_id = metadata.get('workspace_id')

    if not chunks:
        log.warning("workspace.save_embedding_notes.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not note_id or not workspace_id:
        log.warning("workspace.save_embedding_notes.missing_user_id_or_note_id_or_workspace_id", metadata=metadata)
        raise ValueError("Missing user_id or note_id or workspace_id")
    
    try:
        user_uuid = uuid.UUID(str(user_id))
        note_uuid = uuid.UUID(str(note_id))
        workspace_uuid = uuid.UUID(str(workspace_id))
    except ValueError as exc:
        log.error("workspace.notes.save_embedding.invalid_uuid", metadata=metadata, error=str(exc))
        raise ValueError("Invalid UUID string format")

    log.info("workspace.notes.save_embedding.mapping_payloads")
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
        depot.insert_note_chunks(payloads)
    except Exception as exc:
        log.error(
            "workspace.notes.save_embedding.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info("workspace.notes.save_embedding.done", metadata=metadata)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 16 — note paper matcher
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.find_relevant_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def find_relevant_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.find_relevant_papers.start", metadata=metadata)

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")

    if not note_id or not workspace_id:
        log.info("workspace.notes.find_relevant_papers.missing_required_field")
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
            "workspace.notes.find_relevant_papers.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 16 — save matched papers for notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_papers.start")

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    papers = matched_result.get("papers", [])

    if papers:
        log.info("workspace.notes.save_papers.inserting_data", payload_count=len(papers))
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
            depot.insert_note_papers(payloads)
        except Exception as exc:
            log.error(
                "workspace.notes.save_papers.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 16 — save note documents
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_documents",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_documents(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_documents.start")

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info("workspace.notes.save_documents.inserting_data", payload_count=len(similar_chunks))
        payloads: List[NoteDocumentORM] = []

        # payloads enrichment
        for sim in similar_chunks:
            payload = NoteDocumentORM(
                workspace_id=workspace_id,
                user_id=user_id,
                paper_id=sim.get("paper_id"),
                note_id=note_id,
                note_chunk_id=sim.get("chunk_id"),
                note_content=sim.get("chunk_content"),
                document_chunk_id=sim.get("document_id"),
                document_content=sim.get("document_content"),
                document_embedding=sim.get("document_embedding", []),
                similarity_score=sim.get("similarity_score"),
                attributes=sim.get("attributes")
            )
            payloads.append(payload)

        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.insert_note_documents(payloads)
        except Exception as exc:
            log.error(
                "workspace.notes.save_documents.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 16 — dedup daily notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.deduplicate_notes",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def deduplicate_notes(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.deduplicate_notes.start")

    processing_date = metadata.get("processing_date")
    workspace_id = metadata.get("workspace_id")

    if not workspace_id:
        log.warning("workspace.notes.deduplicate_notes.missing_workspace_id", metadata=metadata)
        raise ValueError("Missing workspace_id")

    try:
        clustered_date = datetime.strptime(processing_date, "%Y-%m-%d") if processing_date else datetime.now()
        depot = WorkspaceNoteDepot(db_pool)
        daily_notes = depot.get_chunks_by_workspace(workspace_id)
        notes_ids: List[str] = []
        embeddings: List[List[float]] = []

        for note in daily_notes:
            if note.embedding is not None:
                notes_ids.append(str(note.id))
                embeddings.append(note.embedding)

        embeddings_array = np.array(embeddings)

        ncs = NotesClusteringService()
        result = ncs.find_duplicates(notes_ids=notes_ids, embeddings=embeddings_array)

        if result:
            result_map = {str(item.id): item for item in daily_notes}
            unique = [item for item in daily_notes if str(item.id) in result["unique"]]
            duplicate = {}
            update_payloads: List[NoteChunkORM] = []

            for key, group_ids in result["duplicate"].items():
                key = str(key)
                group = []
                for id_str in group_ids:
                    if id_str in result_map:
                        group.append(result_map[id_str])
                duplicate[key] = group

            metadata["dedup_result"] = {
                "unique": [orm_to_dict(item, exclude={"embedding"}) for item in unique],
                "duplicate": {key: [orm_to_dict(item, exclude={"embedding"}) for item in group] for key, group in duplicate.items()},
            }

            # prepare payloads for updating
            for key, group_ids in result["duplicate"].items():
                for id_str in group_ids:
                    if id_str in result_map:
                        chunk = result_map[id_str]
                        chunk.cluster_label = str(key)
                        chunk.clustered_date = clustered_date
                        update_payloads.append(chunk)

            for u in unique:
                u.cluster_label = "-1"
                u.clustered_date = clustered_date
                update_payloads.append(u)

            # update to database
            depot.update_chunks_with_label(update_payloads)

    except Exception as exc:
        log.error(
            "workspace.notes.deduplicate_notes.failed",
            metadata=metadata,
            error=str(exc),
        )
        raise ValueError("Failed to get chunks for dedup")

    # chain next tasks
    (
        generate_jsonl.s(metadata).set(queue="workspace")
        | process_jsonl.s().set(queue="workspace")
    ).apply_async()

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 of 16 — generate jsonl
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.generate_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("atlazer.celery_app.tasks.workspace.notes.generate_jsonl.start")

    workspace_id = metadata.get("workspace_id")
    language_code = metadata.get("language_code", "en")
    dedup_result = metadata.get("dedup_result", {})
    processing_date = metadata.get("processing_date", "")
    unique = dedup_result.get("unique", [])
    duplicate = dedup_result.get("duplicate", {})
    
    payloads: List[Any] = []
    dkey = processing_date.replace("-", "/")
    ukey = f"{workspace_id}_{dkey}_-1"

    if not workspace_id or not language_code:
        raise ValueError("Missing required ids in metadata")

    for u in unique:
        payload = _build_json(f"{ukey}_{u['note_id']}", u["content"], language_code)
        payloads.append(payload)

    for key, group in duplicate.items():
        combined_content = "\n\n".join([item["content"] for item in group])
        payload = _build_json(f"{workspace_id}_{dkey}_{key}", combined_content, language_code)
        payloads.append(payload)

    key = f"notes/{workspace_id}/{dkey}"
    target_dir = Path(settings.gemini_batch_dir)
    target_file = target_dir / f"{key}.jsonl"
    target_file.parent.mkdir(parents=True, exist_ok=True)

    with open(target_file, "w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # upload file
    file_name = upload_chunk_file(str(target_file), display_name=key)
    if file_name is None:
        raise ValueError(f"Failed to upload file {target_file}")

    log.info(
        "atlazer.celery_app.tasks.workspace.notes.generate_jsonl.done",
        target_file=str(target_file),
        payload_count=len(payloads),
        file_name=file_name,
    )

    metadata["display_name"] = key
    metadata["file_name"] = file_name
    metadata["target_dir"] = str(target_dir)
    metadata["target_file"] = str(target_file)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 9 of 16 — process jsonl file
# ─────────────────────────────────────────────────────────────────────────────#

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.process_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def process_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.process_jsonl.start")

    target_file = metadata.get("target_file")
    if target_file is None:
        raise ValueError("Failed to get target file from metadata")

    file_name = metadata.get("file_name")
    if file_name is None:
        raise ValueError("Failed to get file name from metadata")

    # process to gemini AI
    user_metadata = {
        "key": metadata.get("display_name"),
        "processing_date": metadata.get("processing_date"),
        "workspace_id": metadata.get("workspace_id"),
        "action": "notes_content_enrichment",
    }

    job_name = process_jsonl_file(
        file_name,
        model=settings.gemini_model,
        user_metadata=user_metadata
    )

    if job_name is None:
        raise ValueError(f"Failed to create job for file {target_file}")

    log.info("workspace.notes.process_jsonl.done", job_name=job_name)

    metadata["job_name"] = job_name
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 10 of 16 — save notes content enrichments to database
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_enriched_notes",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_enriched_notes(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_enriched_notes.start")

    now = datetime.now()
    processing_date = metadata.get("processing_date", now.strftime("%Y-%m-%d"))
    key = metadata.get("key", "")  # notes/<workspace_id>/<yyyy>/<mm>/<dd>
    date_value = datetime.strptime(processing_date, "%Y-%m-%d").date()
    workspace_id = metadata.get("workspace_id")
    job_id = metadata.get("job_id")

    if not workspace_id or job_id is None:
        raise ValueError("Failed to get workspace_id or job_id from metadata")

    try:
        depot = WorkspaceNoteDepot(db_pool)
        note_chunks = depot.get_chunks_by_clustered_date(
            date=date_value.isoformat(),
            workspace_id=workspace_id
        )

        log.info(
            "workspace.notes.save_enriched_notes.chunks",
            chunks_count=len(note_chunks)
        )

        # group chunks with cluster_label
        chunk_groups = defaultdict(list)
        for chunk in note_chunks:
            cluster_label = chunk.cluster_label
            chunk_groups[cluster_label].append(orm_to_dict(chunk, exclude={"embedding"}))

        log.info(
            "workspace.notes.save_enriched_notes.chunks_grouped",
            chunks_groups_count=len(chunk_groups)
        )

    except Exception as e:
        log.error("workspace.notes.save_enriched_notes.error", error=str(e))
        raise ValueError(str(e))

    try:
        results = get_batch_results(job_id)
        if results is None:
            raise ValueError("Failed to get results from batch")
    except Exception as e:
        log.error("workspace.notes.save_enriched_notes.error", error=str(e))
        raise ValueError(str(e))

    payloads: List[LearningMaterialNoteORM] = []
    for res in results:
        key = res.get("key", "")  # <workspace_id>/<yyyy>/<mm>/<dd>_<hbdscan_label>_<note_id>
        pattern = r"^([0-9a-f-]{36})_(\d{4}/\d{2}/\d{2})_(-?\d+)(?:_([0-9a-f-]{36}))?$"
        match = re.match(pattern, key)

        if not match:
            log.warning("workspace.notes.save_enriched_notes.invalid_key", key=key)
            continue

        workspace_id, cdate, clabel, note_id = match.groups()

        # map raw chunks
        chunks = chunk_groups[clabel]
        if clabel == '-1' and note_id:
            chunks = [c for c in chunks if c["note_id"] == note_id]

        result = res.get("result", {})
        summary = result.get("summary", None) if isinstance(result, dict) else None
        attributes = res.get("metadata", {})

        payload: LearningMaterialNoteORM = LearningMaterialNoteORM(
            attributes=attributes,
            content=summary,
            chunks=chunks,
            cluster_label=clabel,
            workspace_id=workspace_id,
            clustered_date=date_value,
        )
        payloads.append(payload)

    if payloads:
        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.insert_enriched_notes(payloads)
        except Exception as e:
            log.error("workspace.notes.save_enriched_notes.error", error=str(e))
            raise ValueError(str(e))

    log.info(
        "workspace.notes.save_enriched_notes.batch_results",
        results_count=len(results)
    )

    # process chunking for each enriched notes
    (
        chunk_enriched_notes.s(metadata).set(queue="workspace")
    ).apply_async()

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 11 of 16 — update next processing workspace
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.update_next_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
)
def update_next_processing(
    self,
    results,
    processed_workspaces: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Callback chord: dijalankan otomatis setelah SEMUA task
    deduplicate_notes di grup selesai (sukses).
    `results` = list hasil balik dari tiap task deduplicate_notes.
    """
    log.info(
        "workspace.notes.update_next_processing.start",
        results_count=len(results) if results else 0,
    )

    try:
        now = datetime.now()
        today_midnight = datetime.combine(now.date(), time.min)
        default_next = today_midnight + timedelta(days=1)

        updates = []
        for ws in processed_workspaces:
            current_next_str = ws.get("next_notes_processing_at")

            if current_next_str:
                current_next = datetime.fromisoformat(current_next_str)
                new_next = current_next + timedelta(hours=24)
            else:
                new_next = default_next

            updates.append(
                WorkspaceORM(id=ws.get("id"), next_notes_processing_at=new_next)
            )

        depot = WorkspaceDepot(db_pool)
        depot.update_bulk(updates)

        log.info(
            "workspace.notes.update_next_processing.done",
            updated_count=len(updates),
        )
    except Exception as e:
        log.error("workspace.notes.update_next_processing.error", error=str(e))
        raise ValueError(str(e))

    return {"updated_workspaces_count": len(updates)}


# ─────────────────────────────────────────────────────────────────────────────
# Task 12 of 16 — getting workspace next notes processing
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.process_workspaces",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def process_workspaces(self) -> Dict[str, Any]:
    log.info("workspace.notes.process_workspaces.start")

    metadata: Dict[str, Any] = {}
    processed_workspaces: List[Dict[str, Any]] = []
    now = datetime.now()
    processing_date = now.strftime("%Y-%m-%d")

    try:
        depot = WorkspaceDepot(db_pool)
        workspaces = depot.get_pre_processing_workspaces()

        log.info(
            "workspace.notes.process_workspaces.workspaces",
            workspaces_count=len(workspaces)
        )

        for ws in workspaces:
            processed_workspaces.append(orm_to_dict(ws))

        metadata["workspaces_count"] = len(processed_workspaces)
        if processed_workspaces:
            metadata["processed_workspaces"] = processed_workspaces

        log.info(
            "workspace.notes.process_workspaces.processed_workspaces",
            processed_workspaces=processed_workspaces
        )
    except Exception as e:
        log.error("workspace.notes.process_workspaces.error", error=str(e))
        raise ValueError(str(e))

    if processed_workspaces:
        header = group(
            # run for each workspaces notes
            deduplicate_notes.s(
                metadata={
                    "workspace_id": ws.get("id", None),
                    "language_code": ws.get("language_code", "en"),
                    "processing_date": processing_date,
                },
            ).set(queue="workspace")
            for ws in processed_workspaces
        )

        callback = update_next_processing.s(processed_workspaces).set(queue="workspace")

        job = chord(header)(callback)
        metadata["process_workspaces_job_id"] = job.id

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 13 of 13 — chunk enriched notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.chunk_enriched_notes",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunk_enriched_notes(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.chunk_enriched_notes.start")

    language_code = metadata.get("language_code", "en")
    workspace_id = metadata.get("workspace_id", None)
    processing_date = metadata.get("processing_date", "")
    date_value = datetime.strptime(processing_date, "%Y-%m-%d").date()
    chunks_group: List[List[Dict[str, Any]]] = []

    if not workspace_id:
        log.info("workspace.notes.chunk_enriched_notes.no_workspace")
        raise ValueError("workspace_id is required")

    try:
        depot = WorkspaceNoteDepot(db_pool)
        notes = depot.get_enriched_notes(
            workspace_id=workspace_id,
            clustered_date=date_value
        )
    
        for n in notes:
            chunks = chunk_content(
                text=n.content,
                lang=language_code,
                semantic=True,
                download_models=False,
                embed_model_name=settings.local_embedding_model,
                min_words=settings.context_chunker_min_words,
                max_words=settings.context_chunker_max_words,
            )

            chunks_group.append([
                {
                    "text": chunk, 
                    "metadata": {
                        "material_note_id": str(n.id),
                        "workspace_id": str(workspace_id),
                        "processing_date": processing_date,
                    }
                } for chunk in chunks
            ])

    except Exception as e:
        log.error(
            "workspace.notes.chunk_enriched_notes.error_select",
            error=str(e),
        )
        raise ValueError(str(e))

    if chunks_group:
        # process each chunk independently
        chained_jobs = [
            chain(
                embed_enriched_notes.s(
                    metadata={
                        "workspace_id": workspace_id,
                        "processing_date": processing_date,
                        "chunks": c,
                        "material_note_id": c[0]["metadata"]["material_note_id"] if c else None,
                    }
                ).set(queue="workspace"),
                signature(
                    "atlazer.celery_app.tasks.workspace.notes.save_material_chunks",
                    queue="workspace"
                ),
                signature(
                    "atlazer.celery_app.tasks.workspace.material.find_relevant_papers",
                    queue="workspace"
                ),
                signature(
                    "atlazer.celery_app.tasks.workspace.material.save_documents",
                    queue="workspace"
                )
            ) for c in chunks_group if c
        ]

        if not chained_jobs:
            return metadata

        # process documents after all chunks are processed
        callback = chain(
            signature(
                "atlazer.celery_app.tasks.workspace.material.documents_deduplication",
                args=[
                    {
                        "workspace_id": workspace_id,
                        "processing_date": processing_date,
                        "language_code": language_code,
                    }
                ],
                queue="workspace",
                immutable=True,
            ),
            signature(
                "atlazer.celery_app.tasks.workspace.material.generate_jsonl",
                queue="workspace",
            ),
            signature(
                "atlazer.celery_app.tasks.workspace.material.process_jsonl",
                queue="workspace",
            ),
        )

        chord(group(chained_jobs))(callback)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 14 of 16 — embed_enriched_notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.embed_enriched_notes",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embed_enriched_notes(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.embed_enriched_notes.start", metadata=metadata)

    chunks = metadata.get("chunks", [])
    if not chunks:
        log.info("workspace.notes.embed_enriched_notes.no_chunks")
        raise ValueError("chunks is required")

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "workspace.notes.embed_enriched_notes.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    metadata["chunks"] = embedded_chunks

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 15 of 16 — save chunked materials base on notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_material_chunks",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_material_chunks(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_material_chunks.start")

    chunks = metadata.get("chunks", [])
    workspace_id = metadata.get('workspace_id')

    if not chunks:
        log.warning(
            "workspace.notes.save_material_chunks.no_chunks",
            metadata=metadata
        )
        raise ValueError("No chunks to save")

    try:
        workspace_uuid = uuid.UUID(str(workspace_id))
    except ValueError as exc:
        log.error(
            "workspace.notes.save_material_chunks.invalid_uuid",
            metadata=metadata,
            error=str(exc),
        )
        raise ValueError("Invalid UUID string format")

    payloads: List[LearningMaterialChunkORM] = []
    for chunk in chunks:
        material_note_id = chunk.get("metadata", {}).get('material_note_id')
        try:
            enriched_note_uuid = uuid.UUID(str(material_note_id))
        except ValueError as exc:
            log.error(
                "workspace.notes.save_material_chunks.invalid_uuid",
                metadata=metadata,
                error=str(exc),
            )
            raise ValueError("Invalid UUID string format")

        attributes = {
            "embedding_dim": chunk.get("embedding_dim"),
            "embedding_model": chunk.get("embedding_model"),
            "embedding_adapter": chunk.get("embedding_adapter"),
            "embedding_normalized": chunk.get("embedding_normalized", True),
            "token_count": chunk.get("token_count"),
            "word_count": chunk.get("word_count"),
        }

        payloads.append(LearningMaterialChunkORM(
            material_note_id=enriched_note_uuid,
            workspace_id=workspace_uuid,
            content=chunk.get('text'),
            embedding=chunk.get('embedding'),
            chunk_index=chunk.get('chunk_index'),
            attributes=attributes,
        ))

    if payloads:
        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.insert_material_chunks(payloads)

        except Exception as exc:
            log.error(
                "workspace.notes.save_material_chunks.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info("workspace.notes.save_material_chunks.done", metadata=metadata)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _build_json(key: str, content: str, language_code: str = "en") -> dict:
    return {
        "key": key, 
        "request": {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are a research-context enrichment system.\n"
                                    f"Your task is to transform the provided student notes into structured research contexts that can later be used to retrieve relevant academic papers."
                        },
                        {
                            "text": f"IMPORTANT: \n"
                                    f"This is an ENRICHMENT task, not a research task."
                                    f"\n"
                                    f"You MUST NOT:\n"
                                    f"- add facts that are not present in the notes"
                                    f"- introduce information from your own knowledge"
                                    f"- invent explanations, causes, effects, examples, statistics, theories, authors, papers, or citations"
                                    f"- assume the student's intention beyond what is reasonably expressed"
                                    f"- turn speculation into a fact"
                                    f"- answer questions contained in the notes"
                                    f"- provide conclusions that are not supported by the notes"
                                    f"\n\n"
                                    f"You MAY:\n"
                                    f"- combine semantically similar statements from the provided notes"
                                    f"- remove repetition"
                                    f"- rewrite informal language into clearer academic language"
                                    f"- identify concepts explicitly mentioned or clearly implied by the notes"
                                    f"- formulate neutral research-oriented phrases that preserve the original meaning"
                                    f"- identify uncertainty, disagreement, or questions expressed by the students"
                        },
                        {
                            "text": f"1. Translate and write all the values in the JSON output strictly in the "
                                    f"language corresponding to this language code/name: '{language_code}'. "
                                    "Only translate the values, keep the JSON keys strictly as defined in the schema.\n"
                                    "2. DO NOT use phrases like 'this paper', 'this study', 'the authors', 'this work', 'this document '"
                                    "or any equivalent meta-phrases in the target language. Write the summary directly as "
                                    "factual statements or explanations, completely removing any fluff or context indicating "
                                    "that this is a summary of an academic paper.\n"
                                    "3. PRESERVE TECHNICAL JARGON & INDUSTRY TERMS: Do not translate standard technical terms, "
                                    "academic jargon, widely accepted acronyms, or domain-specific nomenclature (for example: "
                                    "'machine learning', 'zero-shot learning', 'overfitting', 'framework', etc.) if translating "
                                    "them would make the text sound awkward, forced, or lose its precise scientific meaning "
                                    "in the target language. Keep these terms in their original English/technical form."
                                    "4. Do not embed images using Base64, data URIs, or any other encoded image format. If a visual representation is necessary, use inline SVG instead."
                        },
                        {
                            "text": f"SOURCE MATERIAL:\n\n{content}"
                        },
                    ]
                }
            ], 
            "generation_config": {
                "temperature": 0.7,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    }