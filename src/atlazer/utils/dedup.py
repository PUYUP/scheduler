"""
celery_app/utils/dedup.py
──────────────────────────
Redis-backed deduplication for paper IDs.

Two Redis sets are used:
  atlazer_rag:{repository}:queued    – paper is in flight (downloaded / being processed)
  atlazer_rag:{repository}:processed – paper has been fully stored in the vector DB

A paper is skipped on the next Beat run if it exists in either set.
TTL on "queued" prevents stuck papers from blocking re-ingestion forever.
"""

from __future__ import annotations

import redis
import structlog

from typing import cast, Dict, Any
from atlazer.config.settings import settings
from atlazer.celery_app.main import db_pool
from atlazer.storage.progress import ScrapeProgressDepot

log = structlog.get_logger(__name__)

_QUEUED_TTL_SECONDS = 172_800   # 48 h


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[return-value]


def is_already_processed(paper_id: str, repository: str) -> bool:
    """Return True if the paper is queued OR fully processed."""
    r = _get_redis()
    return (
        cast(bool, r.sismember(f"atlazer_rag:{repository}:processed", paper_id))
        or cast(bool, r.sismember(f"atlazer_rag:{repository}:queued", paper_id))
    )


def mark_as_queued(paper_id: str, repository: str) -> None:
    """Mark paper as in-flight.  Expires after 48 h if processing stalls."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.sadd(f"atlazer_rag:{repository}:queued", paper_id)
    pipe.setex(f"atlazer_rag:{repository}:queued:{paper_id}", _QUEUED_TTL_SECONDS, "1")
    pipe.execute()
    log.debug("dedup.queued", paper_id=paper_id, repository=repository)


def mark_as_processed(paper_id: str, repository: str) -> None:
    """
    Promote paper from queued → processed.
    Called by store_paper after successful write.
    """
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem(f"atlazer_rag:{repository}:queued", paper_id)
    pipe.delete(f"atlazer_rag:{repository}:queued:{paper_id}")
    pipe.sadd(f"atlazer_rag:{repository}:processed", paper_id)
    pipe.execute()
    log.debug("dedup.processed", paper_id=paper_id, repository=repository)


def reset_paper(paper_id: str, repository: str) -> None:
    """Force re-ingestion of a specific paper (removes from both sets)."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem(f"atlazer_rag:{repository}:processed", paper_id)
    pipe.srem(f"atlazer_rag:{repository}:queued", paper_id)
    pipe.delete(f"atlazer_rag:{repository}:queued:{paper_id}")
    pipe.execute()
    log.info("dedup.reset", paper_id=paper_id, repository=repository)


def count_processed(repository: str) -> int:
    return cast(int, _get_redis().scard(f"atlazer_rag:{repository}:processed"))


def count_queued(repository: str) -> int:
    return cast(int, _get_redis().scard(f"atlazer_rag:{repository}:queued"))


def is_backfill_complete(topic: str, repository: str, last_position: int) -> bool:
    """True kalau backfill untuk topic+repository ini sudah pernah tuntas."""
    r = _get_redis()
    key = f"backfill:complete:{repository}:{topic}"
    try:
        return cast(bool, r.sismember(key, str(last_position)))
    except Exception as exc:
        r.delete(key)
        log.error(
            "dedup.backfill_complete.error",
            topic=topic,
            repository=repository,
            last_position=last_position,
            error=str(exc)
        )
        return False


def mark_backfill_complete(topic: str, repository: str, last_position: int) -> None:
    """Tandai backfill untuk topic+repository ini sebagai selesai."""
    r = _get_redis()
    key = f"backfill:complete:{repository}:{topic}"
    r.sadd(key, str(last_position))
    log.info(
        "dedup.backfill_complete",
        topic=topic,
        repository=repository,
        last_position=last_position
    )


def check_increment_process(repository: str) -> Dict[str, Any] | None:
    """Return info about topic process.

    Return:
        topic: str
        repository: str
        start: int
        max_results: int
    """
    r = _get_redis()
    key = f"increment:{repository}"
    value = cast(Dict[str, Any], r.hgetall(key))
    if not value:
        return None
    return value


def set_increment_process(repository: str, topic: str, start: int) -> Dict[str, Any]:
    """Set increment process"""
    r = _get_redis()
    key = f"increment:{repository}"
    process = {
        "start": start,
        "topic": topic,
        "repository": repository,
    }

    r.hset(key, mapping=process)

    log.info(
        "dedup.set_increment_process",
        repository=repository,
        topic=topic,
        start=start
    )
    return process


def clear_increment_process(repository: str) -> None:
    """Clear increment process"""
    r = _get_redis()
    key = f"increment:{repository}"
    r.delete(key)
    log.info(
        "dedup.clear_increment_process",
        repository=repository
    )
    return None


def get_topic_start(repository: str, topic: str) -> int:
    """Ambil offset paging untuk topic ini. Default 0 kalau belum pernah diproses."""
    r = _get_redis()
    value = cast(str, r.hget(f"scrape_topic_start:{repository}", topic))
    if value is None:
        # try getting from db
        try:
            depot = ScrapeProgressDepot(db_pool)
            value = str(depot.get_start_offset(repository, topic, 0))
        except Exception as e:
            log.error("dedup.failed_to_get_topic_start_from_db",
                repository=repository,
                topic=topic,
                error=str(e)
            )
            return 0
    
    return int(value) if value is not None else 0


def set_topic_start(repository: str, topic: str, start: int) -> None:
    r = _get_redis()
    r.hset(f"scrape_topic_start:{repository}", topic, str(start))

    # set in db too
    try:
        depot = ScrapeProgressDepot(db_pool)
        depot.set_progress(repository, topic, start)
    except Exception as e:
        log.error("dedup.failed_to_set_topic_start_in_db",
            repository=repository,
            topic=topic,
            start=start,
            error=str(e)
        )
        return


def reset_topic_start(repository: str, topic: str) -> None:
    r = _get_redis()
    r.hdel(f"scrape_topic_start:{repository}", topic)

    # reset in db too
    try:
        depot = ScrapeProgressDepot(db_pool)
        depot.set_progress(repository, topic, 0)
    except Exception as e:
        log.error("dedup.failed_to_reset_topic_start_in_db",
            repository=repository,
            topic=topic,
            error=str(e)
        )
        return
