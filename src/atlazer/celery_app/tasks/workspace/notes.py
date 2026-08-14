from __future__ import annotations

import numpy as np
import uuid
import structlog
import json
import re

from datetime import datetime, date
from typing import Dict, Any, List
from pathlib import Path
from decimal import Decimal
from uuid import UUID
from collections import defaultdict

from atlazer.utils.gemini_batch import upload_chunk_file, process_jsonl_file, get_batch_results
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.workspace_notes import WorkspaceNoteDepot
from atlazer.models.workspace import (
    ChunkNoteMetadata,
    NoteChunkORM,
    NotePaperORM,
    NoteSimilarityORM,
    NoteEnrichmentORM,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 10 — chunk_note
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
# Task 2 of 10 — embed_note
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
# Task 3 of 10 — save embedding note
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
# Task 4 of 10 — note paper matcher
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.match_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def match_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.match_papers.start", metadata=metadata)

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")

    if not note_id or not workspace_id:
        log.info("workspace.notes.match_papers.missing_required_field")
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
            "workspace.notes.match_papers.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 10 — save matched papers for notes
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
# Task 6 of 10 — save note similarities
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_similarities",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_similarities(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_similarities.start")

    note_id = metadata.get("note_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info("workspace.notes.save_similarities.inserting_data", payload_count=len(similar_chunks))
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
            depot.insert_note_similarities(payloads)
        except Exception as exc:
            log.error(
                "workspace.notes.save_similarities.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 10 — dedup daily notes
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.dedup_daily_notes",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def dedup_daily_notes(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.dedup_daily_notes.start")

    workspace_id = metadata.get("workspace_id")
    if not workspace_id:
        log.warning("workspace.notes.dedup_daily_notes.missing_workspace_id", metadata=metadata)
        raise ValueError("Missing workspace_id")

    try:
        now = datetime.now()
        depot = WorkspaceNoteDepot(db_pool)
        daily_notes = depot.get_chunks_daily(workspace_id)
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
                "unique": [_orm_to_dict(item, exclude={"embedding"}) for item in unique],
                "duplicate": {key: [_orm_to_dict(item, exclude={"embedding"}) for item in group] for key, group in duplicate.items()},
            }

            # prepare payloads for updating
            for key, group_ids in result["duplicate"].items():
                for id_str in group_ids:
                    if id_str in result_map:
                        chunk = result_map[id_str]
                        chunk.cluster_label = str(key)
                        chunk.clustered_at = now
                        update_payloads.append(chunk)

            for u in unique:
                u.cluster_label = "-1"
                u.clustered_at = now
                update_payloads.append(u)
            depot.update_chunks_with_label(update_payloads)

    except Exception as exc:
        log.error(
            "workspace.notes.dedup_daily_notes.failed",
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
# Task 8 of 10 — generate jsonl
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
    unique = dedup_result.get("unique", [])
    duplicate = dedup_result.get("duplicate", {})
    
    payloads: List[Any] = []
    now = datetime.now()
    dkey = now.strftime("%Y/%m/%d")
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
# Task 9 of 10 — process jsonl file
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
# Task 10 of 10 — save notes content enrichments to database
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.notes.save_enrichments",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_enrichments(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.notes.save_enrichments.start")

    key = metadata.get("key", "")  # notes/<workspace_id>/<yyyy>/<mm>/<dd>
    parts = key.split("/")
    year = int(parts[2])
    month = int(parts[3])
    day = int(parts[4])
    date_value = date(year, month, day)
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
            "workspace.notes.save_enrichments.chunks",
            chunks_count=len(note_chunks)
        )

        # group chunks with cluster_label
        chunk_groups = defaultdict(list)
        for chunk in note_chunks:
            cluster_label = chunk.cluster_label
            chunk_groups[cluster_label].append(_orm_to_dict(chunk, exclude={"embedding"}))

        log.info(
            "workspace.notes.save_enrichments.chunks_grouped",
            chunks_groups_count=len(chunk_groups)
        )

    except Exception as e:
        log.error("workspace.notes.save_enrichments.error", error=str(e))
        raise ValueError(str(e))

    try:
        results = get_batch_results(job_id)
        if results is None:
            raise ValueError("Failed to get results from batch")
    except Exception as e:
        log.error("workspace.notes.save_enrichments.error", error=str(e))
        raise ValueError(str(e))

    payloads: List[NoteEnrichmentORM] = []
    for res in results:
        key = res.get("key", "")  # <workspace_id>/<yyyy>/<mm>/<dd>_<hbdscan_label>_<note_id>
        pattern = r"^([0-9a-f-]{36})_(\d{4}/\d{2}/\d{2})_(-?\d+)(?:_([0-9a-f-]{36}))?$"
        match = re.match(pattern, key)

        if not match:
            log.warning("workspace.notes.save_enrichments.invalid_key", key=key)
            continue

        workspace_id, cdate, clabel, note_id = match.groups()

        # map raw chunks
        chunks = chunk_groups[clabel]
        if clabel == '-1' and note_id:
            chunks = [c for c in chunks if c["note_id"] == note_id]

        result = res.get("result", {})
        summary = result.get("summary", None) if isinstance(result, dict) else None
        attributes = res.get("metadata", {})

        payload: NoteEnrichmentORM = NoteEnrichmentORM(
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
            depot.insert_note_enrichments(payloads)
        except Exception as e:
            log.error("workspace.notes.save_enrichments.error", error=str(e))
            raise ValueError(str(e))

    log.info(
        "workspace.notes.save_enrichments.batch_results",
        results_count=len(results)
    )

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _orm_to_dict(obj, exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    result = {}
    for c in obj.__table__.columns:
        if c.name in exclude:
            continue
        value = getattr(obj, c.name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, (datetime, date)):
            value = value.isoformat()
        elif isinstance(value, UUID):
            value = str(value)
        elif isinstance(value, Decimal):
            value = float(value)
        result[c.name] = value
    return result


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