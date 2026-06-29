"""
celery_app/utils/dedup.py
──────────────────────────
Redis-backed deduplication for paper IDs.

Two Redis sets are used:
  curiosift_rag:{repository}:queued    – paper is in flight (downloaded / being processed)
  curiosift_rag:{repository}:processed – paper has been fully stored in the vector DB

A paper is skipped on the next Beat run if it exists in either set.
TTL on "queued" prevents stuck papers from blocking re-ingestion forever.
"""

import redis
import structlog

from __future__ import annotations
from typing import cast
from config.settings import settings

log = structlog.get_logger(__name__)

_QUEUED_TTL_SECONDS = 172_800   # 48 h


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[return-value]


def is_already_processed(paper_id: str, repository: str) -> bool:
    """Return True if the paper is queued OR fully processed."""
    r = _get_redis()
    return (
        cast(bool, r.sismember(f"curiosift_rag:{repository}:processed", paper_id))
        or cast(bool, r.sismember(f"curiosift_rag:{repository}:queued", paper_id))
    )


def mark_as_queued(paper_id: str, repository: str) -> None:
    """Mark paper as in-flight.  Expires after 48 h if processing stalls."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.sadd(f"curiosift_rag:{repository}:queued", paper_id)
    pipe.setex(f"curiosift_rag:{repository}:queued:{paper_id}", _QUEUED_TTL_SECONDS, "1")
    pipe.execute()
    log.debug("dedup.queued", paper_id=paper_id, repository=repository)


def mark_as_processed(paper_id: str, repository: str) -> None:
    """
    Promote paper from queued → processed.
    Called by store_chunks after successful write.
    """
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem(f"curiosift_rag:{repository}:queued", paper_id)
    pipe.delete(f"curiosift_rag:{repository}:queued:{paper_id}")
    pipe.sadd(f"curiosift_rag:{repository}:processed", paper_id)
    pipe.execute()
    log.debug("dedup.processed", paper_id=paper_id, repository=repository)


def reset_paper(paper_id: str, repository: str) -> None:
    """Force re-ingestion of a specific paper (removes from both sets)."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem(f"curiosift_rag:{repository}:processed", paper_id)
    pipe.srem(f"curiosift_rag:{repository}:queued", paper_id)
    pipe.delete(f"curiosift_rag:{repository}:queued:{paper_id}")
    pipe.execute()
    log.info("dedup.reset", paper_id=paper_id, repository=repository)


def count_processed(repository: str) -> int:
    return cast(int, _get_redis().scard(f"curiosift_rag:{repository}:processed"))


def count_queued(repository: str) -> int:
    return cast(int, _get_redis().scard(f"curiosift_rag:{repository}:queued"))
