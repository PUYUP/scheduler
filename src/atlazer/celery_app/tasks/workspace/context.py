from __future__ import annotations

import uuid
import structlog
import json

from pathlib import Path
from typing import Dict, Any, List

from atlazer.utils.gemini_batch import upload_chunk_file, process_jsonl_file, get_batch_results
from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_content
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.workspace.context import WorkspaceContextDepot
from atlazer.models.workspace import (
    ChunkContextMetadata,
    ContextChunkORM,
    ContextPaperORM,
    ContextDocumentORM,
    ContextPaperSummaryORM,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 10 — chunk_context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.chunking",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunking(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    validated = ChunkContextMetadata.model_validate(metadata)
    content = validated.content
    language_code = validated.language_code

    log.info("workspace.context.chunking.start", metadata=validated.model_dump())

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
        "workspace.context.chunking.done",
        chunk_count=len(validated.chunks),
        metadata=validated.model_dump()
    )

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 10 — embed_context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.embedding",
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
    log.info("workspace.context.embedding.start", metadata=metadata)

    if not chunks:
        raise ValueError("No chunks to embed")

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "workspace.context.embedding.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "workspace.context.embedding.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 10 — save embedding context
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.save_embedding",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.save_embedding.start")

    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    context_id = metadata.get('context_id')
    workspace_id = metadata.get('workspace_id')

    if not chunks:
        log.warning("workspace.context.save_embedding.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not context_id or not workspace_id:
        log.warning("workspace.context.save_embedding.missing_user_id_or_context_id_or_workspace_id", metadata=metadata)
        raise ValueError("Missing user_id or context_id or workspace_id")
    
    try:
        user_uuid = uuid.UUID(str(user_id))
        context_uuid = uuid.UUID(str(context_id))
        workspace_uuid = uuid.UUID(str(workspace_id))
    except ValueError as exc:
        log.error(
            "workspace.context.save_embedding.invalid_uuid",
            metadata=metadata,
            error=str(exc),
        )
        raise ValueError("Invalid UUID string format")

    log.info("workspace.context.save_embedding.mapping_payloads")
    payloads: List[ContextChunkORM] = []
    for chunk in chunks:
        attributes = {
            "embedding_dim": chunk.get("embedding_dim"),
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

    try:
        depot = WorkspaceContextDepot(db_pool)
        depot.insert_context_chunks(payloads)
    except Exception as exc:
        log.error(
            "workspace.context.save_embedding.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info("workspace.context.save_embedding.done", metadata=metadata)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 10 — context paper matcher
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.find_relevant_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def find_relevant_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.find_relevant_papers.start", metadata=metadata)

    context_id = metadata.get("context_id")
    workspace_id = metadata.get("workspace_id")

    if not context_id or not workspace_id:
        log.info("workspace.context.missing_required_field")
        raise ValueError("Missing context_id or workspace_id")

    try:
        depot = WorkspaceContextDepot(db_pool)
        chunks = depot.get_chunks_by_context_id(context_id)
        matcher = depot.match_context_with_papers(chunks=chunks)

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
            "workspace.context.find_relevant_papers.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata

# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 10 — save mathced papers
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.save_papers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_papers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.save_papers.start")

    context_id = metadata.get("context_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    papers = matched_result.get("papers", [])

    if papers:
        log.info("workspace.context.save_papers.inserting_data", payload_count=len(papers))
        payloads: List[ContextPaperORM] = []

        # payloads enrichment
        for paper in papers:
            payload = ContextPaperORM(
                paper_id=paper.get("id"),
                context_id=context_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            payloads.append(payload)

        try:
            depot = WorkspaceContextDepot(db_pool)
            depot.insert_context_papers(payloads)
        except Exception as exc:
            log.error(
                "workspace.context.save_papers.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 10 — save context documents
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.save_documents",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_documents(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.save_documents.start")

    context_id = metadata.get("context_id")
    workspace_id = metadata.get("workspace_id")
    user_id = metadata.get("user_id")
    matched_result = metadata.get("matched_result", {})
    similar_chunks = matched_result.get("similar_chunks", [])

    if similar_chunks:
        log.info("workspace.context.save_documents.inserting_data", payload_count=len(similar_chunks))
        payloads: List[ContextDocumentORM] = []

        # payloads enrichment
        for sim in similar_chunks:
            payload = ContextDocumentORM(
                workspace_id=workspace_id,
                user_id=user_id,
                paper_id=sim.get("paper_id"),
                context_id=context_id,
                context_chunk_id=sim.get("chunk_id"),
                context_content=sim.get("chunk_content"),
                document_chunk_id=sim.get("document_id"),
                document_content=sim.get("document_content"),
                document_embedding=sim.get("document_embedding", []),
                similarity_score=sim.get("similarity_score"),
                attributes=sim.get("attributes")
            )
            payloads.append(payload)

        try:
            depot = WorkspaceContextDepot(db_pool)
            depot.insert_context_documents(payloads)
        except Exception as exc:
            log.error(
                "workspace.context.save_documents.failed",
                metadata=metadata,
                error=str(exc),
                attempt=self.request.retries,
            )
            raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 10 — summarize similar chunks -> save as paper summary
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.summarize_similarities",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def summarize_similarities(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.summarize_similarities.start")
    context_id = metadata.get("context_id")

    if not context_id:
        log.error("workspace.context.summarize_similarities.missing_context_id")
        raise ValueError("Missing context_id")

    try:
        depot = WorkspaceContextDepot(db_pool)
        similarities = depot.get_documents_by_context_id(context_id)
        similarities_papers: List[Dict[str, Any]] = []

        for paper_id in similarities:
            p = similarities[paper_id]
            paper: Dict[str, Any] = {
                "paper_id": paper_id,
                "document_contents": []
            }
            for c in p:
                paper["document_contents"].append(c.document_content)
            similarities_papers.append(paper)

        if similarities_papers:
            metadata["similarities_papers"] = similarities_papers
    except Exception as exc:
        log.error(
            "workspace.context.summarize_similarities.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    # chain next tasks
    (
        generate_jsonl.s(metadata).set(queue="workspace")
        | process_jsonl.s().set(queue="workspace")
    ).apply_async()

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 of 10 — generate jsonl base for summarizing similar papers
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.generate_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("atlazer.celery_app.tasks.workspace.context.generate_jsonl.start")

    context_id = metadata.get("context_id")
    workspace_id = metadata.get("workspace_id")
    language_code = metadata.get("language_code", "en")
    similarities_papers = metadata.get("similarities_papers", [])
    payloads: List[Any] = []

    if not workspace_id or not context_id or not language_code:
        raise ValueError("Missing required ids in metadata")

    for sim in similarities_papers:
        paper_id = sim["paper_id"]
        document = "\n\n".join(sim["document_contents"])
        payload = {
            "key": f"{workspace_id}_{context_id}_{paper_id}", 
            "request": {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "You are a scientific research assistant."
                            },
                            {
                                "text": f"Analyze the provided excerpts from a scientific paper based on the given context. First, infer the main research topic or field represented by the excerpts. Then, summarize only the key findings, concepts, methods, or evidence that are explicitly supported by the provided excerpts and relevant to the context. Briefly explain why the excerpts are relevant to the context, using only evidence found in the excerpts.\n"
                                        f"Do not assume, invent, or infer specific facts that are not supported by the excerpts. Do not present general knowledge, speculation, or assumptions as findings from the paper. If the excerpts are insufficient to determine something, explicitly state that the available excerpts do not provide enough information. Do not treat these excerpts as representative of the entire paper.\n"
                                        f"Use cautious language when making interpretations, such as 'the excerpts suggest' or 'the provided excerpts indicate', when the evidence is indirect or incomplete. Use Markdown formatting when it improves clarity or organization."
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
                                "text": document
                            }
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

        payloads.append(payload)

    key = f"contexts/{workspace_id}/{context_id}"
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
        "atlazer.celery_app.tasks.workspace.context.generate_jsonl.done",
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
    name="atlazer.celery_app.tasks.workspace.context.process_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def process_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.process_jsonl.start")

    target_file = metadata.get("target_file")
    if target_file is None:
        raise ValueError("Failed to get target file from metadata")

    file_name = metadata.get("file_name")
    if file_name is None:
        raise ValueError("Failed to get file name from metadata")

    # process to gemini AI
    user_metadata = {
        "language_code": metadata.get("language_code", "en"),
        "context_id": metadata.get("context_id"),
        "workspace_id": metadata.get("workspace_id"),
        "action": "context_papers_summary_generation",
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
# Task 10 of 10 — save paper summaries to database
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.workspace.context.save_summaries",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="workspace",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_summaries(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("workspace.context.save_summaries.start")

    context_id = metadata.get("context_id")
    workspace_id = metadata.get("workspace_id")
    job_id = metadata.get("job_id")
    if job_id is None:
        raise ValueError("Failed to get job id from metadata")

    try:
        results = get_batch_results(job_id)
        if results is None:
            raise ValueError("Failed to get results from batch")
    except Exception as e:
        log.error("workspace.context.save_summaries.error", error=str(e))
        raise ValueError(str(e))

    log.info(
        "workspace.context.save_summaries.batch_results",
        results_count=len(results)
    )

    payloads: List[ContextPaperSummaryORM] = []

    for res in results:
        key = res.get("key", None)  # <workspace_id>_<context_id>_<paper_id>
        if key:
            paper_id = key.split("_")[-1]
            result = res.get("result", {})
            summary = result.get("summary", None) if isinstance(result, dict) else None
            attributes = res.get("metadata", {})

            payload: ContextPaperSummaryORM = ContextPaperSummaryORM(
                workspace_id=workspace_id,
                context_id=context_id,
                paper_id=paper_id,
                content=summary,
                attributes=attributes
            )
            payloads.append(payload)

    if payloads:
        try:
            depot = WorkspaceContextDepot(db_pool)
            depot.insert_paper_summaries(payloads)
        except Exception as e:
            log.error("workspace.context.save_summaries.error", error=str(e))
            raise ValueError(str(e))

    log.info("workspace.context.save_summaries.success", metadata=metadata)
    return metadata
