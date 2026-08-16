from __future__ import annotations

import numpy as np
import structlog
import json
import re

from datetime import datetime, date
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
from celery import group

from atlazer.utils.text_cleaner import orm_to_dict
from atlazer.utils.gemini_batch import upload_chunk_file, process_jsonl_file, get_batch_results
from atlazer.utils.notes_clustering import NotesClusteringService
from atlazer.celery_app.main import app, db_pool
from atlazer.config.settings import settings
from atlazer.storage.workspace.notes import WorkspaceNoteDepot
from atlazer.storage.workspace.material import LearningMaterialDepot
from atlazer.models.workspace import (
    LearningMaterialDocumentORM,
    LearningMaterialSourceORM,
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
    log.info("workspace.material.find_relevant_papers.start")

    workspace_id = metadata.get("workspace_id")
    material_note_id = metadata.get("material_note_id")

    if not material_note_id or not workspace_id:
        log.info("workspace.material.find_relevant_papers.missing_required_field")
        raise ValueError("Missing material_note_id or workspace_id")

    try:
        depot = WorkspaceNoteDepot(db_pool)
        chunks = depot.get_enriched_chunks(workspace_id=workspace_id, material_note_id=material_note_id)
        matcher = depot.match_note_with_papers(chunks=chunks, candidate_pool_size=100)

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
# Task 2 of 10 — save save documents
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.save_documents",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_documents(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.save_documents.start")

    material_note_id = metadata.get("material_note_id")
    workspace_id = metadata.get("workspace_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info(
            "workspace.material.save_documents.inserting_data",
            payload_count=len(similar_chunks)
        )

        # payloads enrichment
        payloads: List[LearningMaterialDocumentORM] = []

        for sim in similar_chunks:
            payload = LearningMaterialDocumentORM(
                workspace_id=workspace_id,
                material_note_id=material_note_id,
                material_chunk_id=sim.get("chunk_id"),
                paper_id=sim.get("paper_id"),
                material_note_content=sim.get("chunk_content", ""),
                document_chunk_id=sim.get("document_id"),
                document_content=sim.get("document_content", ""),
                document_embedding=sim.get("document_embedding", []),
                similarity_score=sim.get("similarity_score", 0.0),
                attributes=sim.get("attributes", {})
            )
            payloads.append(payload)

        try:
            depot = WorkspaceNoteDepot(db_pool)
            depot.insert_material_documents(payloads)
        except Exception as exc:
            log.error(
                "workspace.material.save_documents.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 10 — deduplicate the material-document similarities
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.documents_deduplication",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def documents_deduplication(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.documents_deduplication.start")

    workspace_id = metadata.get("workspace_id")
    processing_date = metadata.get("processing_date")

    if not workspace_id or not processing_date:
        log.warning("workspace.material.documents_deduplication.missing_required_field")
        raise ValueError("Missing workspace_id or processing_date")

    try:
        depot = LearningMaterialDepot(db_pool)
        documents = depot.get_documents(workspace_id, processing_date)
        document_count = len(documents)

        if document_count > 0:
            # deduplication process
            log.info(
                "workspace.material.documents_deduplication.processing",
                document_count=document_count
            )

            document_ids: List[str] = []
            embeddings: List[List[float]] = []

            for doc in documents:
                if doc.document_embedding is not None:
                    document_ids.append(str(doc.id))
                    embeddings.append(doc.document_embedding)

            embeddings_array = np.array(embeddings)

            ncs = NotesClusteringService()
            result = ncs.find_duplicates(notes_ids=document_ids, embeddings=embeddings_array)

            if result:
                result_map = {str(item.id): item for item in documents}
                unique = [item for item in documents if str(item.id) in result["unique"]]
                duplicate = {}
                update_payloads: List[LearningMaterialDocumentORM] = []
                clustered_date_obj = date.fromisoformat(processing_date)

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
                            chunk.clustered_date = clustered_date_obj
                            update_payloads.append(chunk)

                for u in unique:
                    u.cluster_label = "-1"
                    u.clustered_date = clustered_date_obj
                    update_payloads.append(u)

                # update to database
                depot.update_documents_with_label(update_payloads)

        metadata["document_count"] = document_count
    except Exception as exc:
        log.error(
            "workspace.material.documents_deduplication.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 10 — generate jsonl
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.generate_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("atlazer.celery_app.tasks.workspace.material.generate_jsonl.start")

    workspace_id = metadata.get("workspace_id")
    language_code = metadata.get("language_code", "en")
    processing_date = metadata.get("processing_date", "")

    dedup_result = metadata.get("dedup_result", {})
    unique = dedup_result.get("unique", [])
    duplicate = dedup_result.get("duplicate", {})

    payloads: List[Any] = []
    dkey = processing_date.replace("-", "/")
    ukey = f"{workspace_id}_{dkey}_-1"

    if not workspace_id or not language_code:
        raise ValueError("Missing required ids in metadata")

    for u in unique:
        payload = _build_json(f"{ukey}_{u['material_note_id']}", u["document_content"], language_code)
        payloads.append(payload)

    for key, group in duplicate.items():
        combined_content = "\n\n".join([item["document_content"] for item in group])
        payload = _build_json(f"{workspace_id}_{dkey}_{key}", combined_content, language_code)
        payloads.append(payload)

    key = f"materials/{workspace_id}/{dkey}"
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
        "atlazer.celery_app.tasks.workspace.material.generate_jsonl.done",
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
# Task 5 of 10 — process jsonl file
# ─────────────────────────────────────────────────────────────────────────────#

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.process_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def process_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.process_jsonl.start")

    target_file = metadata.get("target_file")
    if target_file is None:
        raise ValueError("Failed to get target file from metadata")

    file_name = metadata.get("file_name")
    if file_name is None:
        raise ValueError("Failed to get file name from metadata")

    # process to gemini AI
    user_metadata = {
        "processing_date": metadata.get("processing_date"),
        "workspace_id": metadata.get("workspace_id"),
        "action": "material_build_sources",
    }

    job_name = process_jsonl_file(
        file_name,
        model=settings.gemini_model,
        user_metadata=user_metadata
    )

    if job_name is None:
        raise ValueError(f"Failed to create job for file {target_file}")

    log.info("workspace.context.process_jsonl.done", job_name=job_name)

    metadata["job_name"] = job_name
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 10 — build sources
# ─────────────────────────────────────────────────────────────────────────────#

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.build_sources",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def build_sources(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.build_sources.start")

    now = datetime.now()
    processing_date = metadata.get("processing_date", now.strftime("%Y-%m-%d"))
    key = metadata.get("key", "")  # notes/<workspace_id>/<yyyy>/<mm>/<dd>
    date_value = datetime.strptime(processing_date, "%Y-%m-%d").date()
    workspace_id = metadata.get("workspace_id")
    job_id = metadata.get("job_id")

    if not workspace_id or job_id is None:
        raise ValueError("Failed to get workspace_id or job_id from metadata")

    try:
        depot = LearningMaterialDepot(db_pool)
        documents = depot.get_documents(
            workspace_id=workspace_id,
            processing_date=date_value.strftime("%Y-%m-%d")
        )

        log.info(
            "workspace.material.build_sources.documents",
            documents_count=len(documents)
        )

        # group chunks with cluster_label
        document_groups = defaultdict(list)
        for document in documents:
            cluster_label = document.cluster_label
            document_groups[cluster_label].append(orm_to_dict(document, exclude={"embedding"}))

        log.info(
            "workspace.material.build_sources.documents_grouped",
            documents_groups_count=len(document_groups)
        )

    except Exception as e:
        log.error("workspace.material.build_sources.error", error=str(e))
        raise ValueError(str(e))

    try:
        results = get_batch_results(job_id)
        if results is None:
            raise ValueError("Failed to get results from batch")
    except Exception as e:
        log.error("workspace.material.build_sources.error", error=str(e))
        raise ValueError(str(e))

    payloads: List[LearningMaterialSourceORM] = []
    for res in results:
        key = res.get("key", "")  # <workspace_id>/<yyyy>/<mm>/<dd>_<hbdscan_label>_<note_id>
        pattern = r"^([0-9a-f-]{36})_(\d{4}/\d{2}/\d{2})_(-?\d+)(?:_([0-9a-f-]{36}))?$"
        match = re.match(pattern, key)

        if not match:
            log.warning("workspace.material.build_sources.invalid_key", key=key)
            continue

        workspace_id, cdate, clabel, material_note_id = match.groups()

        # map raw chunks
        documents = document_groups[clabel]
        if clabel == '-1' and material_note_id:
            documents = [c for c in documents if c["material_note_id"] == material_note_id]

        # remove embedding, to large
        documents = [
            {k: v for k, v in doc.items() if k != "document_embedding"}
            for doc in documents
        ]

        result = res.get("result", {})
        summary = result.get("summary", None) if isinstance(result, dict) else None
        attributes = res.get("metadata", {})

        payload: LearningMaterialSourceORM = LearningMaterialSourceORM(
            attributes=attributes,
            content=summary,
            chunks=documents,
            cluster_label=clabel,
            workspace_id=workspace_id,
            clustered_date=date_value,
        )
        payloads.append(payload)

    if payloads:
        try:
            depot = LearningMaterialDepot(db_pool)
            depot.insert_sources(payloads)
        except Exception as e:
            log.error("workspace.material.build_sources.error", error=str(e))
            raise ValueError(str(e))

    log.info(
        "workspace.material.build_sources.batch_results",
        results_count=len(results)
    )

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 10 — build material
# Combine all learning_sources into single PDF file
# ─────────────────────────────────────────────────────────────────────────────#

@app.task(
    name="atlazer.celery_app.tasks.workspace.material.build_material",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def build_material(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.material.build_material.start")

    try:
        depot = LearningMaterialDepot(db_pool)
    except Exception as e:
        log.error("workspace.material.build_material.error", error=str(e))
        raise ValueError(str(e))

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
