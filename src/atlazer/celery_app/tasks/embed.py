"""
tasks/embed.py
──────────────────────────
Tier-3 tasks (queue: embed)
─────────────────────────────────────────────────────────────────────────
Flow (continued from process.py):
  chunk_document → generate_embeddings → store_chunks
─────────────────────────────────────────────────────────────────────────
Design notes:
  • Batching     – chunks are grouped into batches of `embedding_batch_size`
                   before hitting the API to minimise round-trips.
  • Provider     – "openai"  → text-embedding-3-small (default)
                   "local"   → sentence-transformers (BAAI/bge-small-en)
  • Rate limit   – the `rate_limit` annotation (20/m) in celery_config
                   keeps us under the OpenAI embeddings quota.
  • store_chunks – intentionally thin: just attaches the vectors to each
                   chunk dict and hands off to your database layer.
                   Swap in your pgvector / Qdrant / Weaviate writer here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import structlog

from atlazer.celery_app.main import app
from atlazer.models.paper import PaperCreate
from atlazer.config.settings import settings
from atlazer.storage.paper import PaperDepot
from atlazer.celery_app.main import db_pool
from atlazer.models.document import DocumentChunkCreate
from atlazer.utils.embedder import chunks_to_vector

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 7 — generate_embeddings
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.embed.generate_embeddings",
    bind=True,
    max_retries=10,
    default_retry_delay=30,
    queue="embed",
    # rate_limit set globally in celery_config.py  →  "20/m"
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_embeddings(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates embedding vectors for every chunk in `metadata["chunks"]`.

    Chunks are batched to stay within API token limits.
    Each chunk is enriched with an `embedding` key (list[float]).

    Returns the full metadata dict with embedded chunks.
    """
    paper_id    = metadata["paper_id"]
    repository  = metadata["repository"]
    chunks      = metadata.get("chunks", [])
    embedded_chunks: List[Dict[str, Any]] = []

    if not chunks:
        log.warning("generate_embeddings.no_chunks", paper_id=paper_id, repository=repository)
        return metadata

    log.info(
        "generate_embeddings.start",
        paper_id=paper_id,
        repository=repository,
        chunks=len(chunks),
        provider=settings.embedding_provider,
    )

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "generate_embeddings.failed",
            paper_id=paper_id,
            repository=repository,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "generate_embeddings.done",
        paper_id=paper_id,
        repository=repository,
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 7 — store_chunks
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.embed.store_chunks",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue="embed",
    ignore_result=False,
)
def store_chunks(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists embedded chunks to your vector store.

    ┌──────────────────────────────────────────────────────────┐
    │  INTEGRATION POINT — replace the body of _write_chunks   │
    │  with your actual DB writer (pgvector, Qdrant, Pinecone, │
    │  Weaviate, Chroma …).                                     │
    └──────────────────────────────────────────────────────────┘

    Each chunk arriving here is guaranteed to have:
      chunk_id, paper_id, title, section, text, embedding,
      embedding_model, embedding_dim, authors, categories, 
      published, doi, token_count
    """
    paper_id    = metadata["paper_id"]
    repository  = metadata["repository"]
    chunks      = metadata.get("chunks", [])

    # Drop chunks that somehow lost their embedding
    valid_chunks = [c for c in chunks if c.get("embedding")]
    if not valid_chunks:
        log.warning("store_chunks.no_valid_chunks", paper_id=paper_id, repository=repository)
        return {"paper_id": paper_id, "repository": repository, "stored": 0, "status": "skipped"}

    log.info("store_chunks.start", paper_id=paper_id, repository=repository, count=len(valid_chunks))

    try:
        paper_uuid = _write_paper(metadata)
    except Exception as exc:
        log.error("store_paper.failed", paper_id=paper_id, repository=repository, error=str(exc))
        raise self.retry(exc=exc)

    try:
        stored_count = _write_chunks(paper_uuid, valid_chunks)
    except Exception as exc:
        log.error("store_chunks.failed", paper_id=paper_id, repository=repository, error=str(exc))
        raise self.retry(exc=exc)

    # Mark this paper as fully processed so scrape_topic won't re-queue it
    from atlazer.utils.dedup import mark_as_processed
    mark_as_processed(paper_id, repository=repository)

    # Clean up the local PDF to reclaim disk space
    _cleanup_pdf(metadata.get("local_pdf_path"))

    # Clean up the GROBID output
    out_dir = Path(metadata["local_pdf_path"]).parent / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    log.info("store_chunks.done", paper_id=paper_id, repository=repository, stored=stored_count)

    return {
        "paper_id":   paper_id,
        "repository": repository,
        "stored":     stored_count,
        "status":     "ok",
        "title":      metadata.get("title", ""),
        "categories": metadata.get("categories", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration stub — swap this out for your real vector store
# ─────────────────────────────────────────────────────────────────────────────

def _write_paper(metadata: Dict[str, Any]) -> str:
    # Collect paper data
    published = metadata.get('published', None)
    year = None
    if published:
        # parse date string to datetime object
        published = datetime.fromisoformat(published)
        year = published.year
        
    payload = PaperCreate(
        title=metadata.get('title', ''),
        abstract=metadata.get('abstract', ''),
        authors=metadata.get('authors', []),
        identifier=metadata.get('paper_id', ''),
        repository=metadata.get('repository', ''),
        doi=metadata.get('doi', None),
        date_published=published.strftime("%Y-%m-%d") if published else None,
        pdf_url=metadata.get('pdf_url', ''),
        processing_status='done',
        keywords=metadata.get('keywords', []),
        fields_of_study=metadata.get('categories', []),
        year=year,
        open_access=True,
        processing_tool='grobid',
        processing_version='0.9.0',
    )

    paper_depot = PaperDepot(db_pool)
    return paper_depot.upsert_paper(payload)


def _write_chunks(paper_uuid: str, chunks: List[Dict[str, Any]]) -> int:
    """
    Write `chunks` to your vector database.

    Each chunk dict has this shape:
    {
        "chunk_id":        "2401.12345_2_0",
        "paper_id":        "2401.12345",
        "repository":      "arxiv",
        "title":           "Attention Is All You Need",
        "section":         "Methods",
        "text":            "We propose a new...",
        "embedding":       [0.012, -0.034, ...],   # list[float]
        "embedding_model": "text-embedding-3-small",
        "embedding_dim":   1536,
        "authors":         ["Vaswani, A.", ...],
        "categories":      ["cs.CL", "cs.LG"],
        "published":       "2017-06-12T00:00:00+00:00",
        "doi":             "...",
        "token_count":     420,
    }
    """

    payloads: List[DocumentChunkCreate] = []

    for chunk in chunks:
        content = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', '')
        section_orders = chunk_id.rsplit('_', 2)[-2:]
    
        payloads.append(
            DocumentChunkCreate(
                paper_id=paper_uuid,
                repository=chunk.get("repository", ""),
                identifier=chunk.get("paper_id", ""),
                section=chunk.get('section', ''),
                section_order='_'.join(section_orders),
                chunk=chunk_id,
                chunk_type='body',
                content=content,
                word_count=len(content.split()),
                embedding=chunk.get('embedding', []),
                embedding_model=chunk.get('embedding_model', None),
                token_count=chunk.get('token_count', None),
            )
        )

    paper_depot = PaperDepot(db_pool)
    paper_depot.bulk_insert_chunks(payloads)

    # ── Default: log-only (replace with real writer) ──────────────────
    log.info(
        "_write_chunks.stub",
        count=len(chunks),
        sample_chunk_id=chunks[0]["chunk_id"] if chunks else None,
        note="Replace _write_chunks() with your vector store writer",
    )
    return len(chunks)


def _cleanup_pdf(pdf_path: str | None) -> None:
    """Remove the local PDF after successful embedding to free disk space."""
    if not pdf_path:
        return
    from pathlib import Path
    p = Path(pdf_path)
    if p.exists():
        p.unlink()
        log.debug("store_chunks.pdf_deleted", path=str(p))