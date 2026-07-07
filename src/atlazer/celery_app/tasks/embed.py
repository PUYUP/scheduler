"""
tasks/embed.py
──────────────────────────
Tier-3 tasks (queue: embed)
─────────────────────────────────────────────────────────────────────────
Flow (continued from process.py):
  chunk_document → generate_embeddings → store_paper
─────────────────────────────────────────────────────────────────────────
Design notes:
  • Batching     – chunks are grouped into batches of `embedding_batch_size`
                   before hitting the API to minimise round-trips.
  • Provider     – "local"   → sentence-transformers (BAAI/bge-small-en)
  • Rate limit   – the `rate_limit` annotation (20/m) in celery_config
                   keeps us under the OpenAI embeddings quota.
  • store_paper  – intentionally thin: just attaches the vectors to each
                   chunk dict and hands off to your database layer.
                   Swap in your pgvector / Qdrant / Weaviate writer here.
"""

from __future__ import annotations
from typing import Any, Dict, List

import structlog

from celery import signature
from atlazer.celery_app.main import app
from atlazer.config.settings import settings
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

    # Chain: embed_chunks → store_paper (store queue)
    signature(
        "atlazer.celery_app.tasks.store.store_paper",
        args=(metadata,),
        queue="store",
        immutable=False,
    ).apply_async()

    return metadata
