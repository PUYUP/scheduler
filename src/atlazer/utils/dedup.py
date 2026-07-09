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

from typing import cast, Dict, Any, List, Optional
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


def _decode(value: Any) -> Any:
    """Redis bisa mengembalikan bytes kalau client tidak dikonfigurasi
    `decode_responses=True`. `cast()` di kode lama TIDAK benar-benar
    mengonversi apapun -- itu murni type hint untuk mypy, runtime value
    tetap bytes kalau client belum decode_responses. Helper ini defensif
    terhadap kedua kemungkinan, apapun konfigurasi client-nya.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
 
 
def _decode_mapping(mapping: Dict[Any, Any]) -> Dict[str, str]:
    return {_decode(k): _decode(v) for k, v in mapping.items()}
 
 
def check_increment_process(repository: str) -> Optional[Dict[str, Any]]:
    """Return info about topic process.
 
    Return:
        topic: str
        repository: str
        start: int
    """
    r = _get_redis()
    key = f"increment:{repository}"
    # NOTE: redis-py men-share signature command sync & async, jadi hgetall()
    # di-type sebagai Union[Awaitable[dict], dict] walau kita pakai client
    # sync di sini (tidak ada `await` di file ini sama sekali). cast() di
    # bawah cuma menyingkirkan cabang Awaitable yang secara runtime memang
    # tidak mungkin terjadi -- ini beda dari cast() yang salah di kode lama,
    # yang dulu dipakai untuk mengklaim ISI data bytes sudah jadi str
    # (itu tidak benar, decode manual tetap perlu, lihat _decode_mapping()).
    raw = cast(Dict[Any, Any], r.hgetall(key))
    if not raw:
        return None
    return _decode_mapping(raw)
 
 
def set_increment_process(repository: str, topic: str, start: int) -> Dict[str, Any]:
    """Set increment process"""
    r = _get_redis()
    key = f"increment:{repository}"
    process = {
        "start": str(start),
        "topic": topic,
        "repository": repository,
    }
 
    r.hset(key, mapping=process)
 
    log.info(
        "dedup.set_increment_process",
        repository=repository,
        topic=topic,
        start=start,
    )
    return process
 
 
def clear_increment_process(repository: str) -> None:
    """Clear increment process"""
    r = _get_redis()
    key = f"increment:{repository}"
    r.delete(key)
    log.info(
        "dedup.clear_increment_process",
        repository=repository,
    )
    return None
 
 
def claim_next_topic(repository: str, serving_topics: List[str]) -> Dict[str, Any]:
    """Klaim topic yang harus diproses SEKARANG, dan langsung majukan pointer
    round-robin ke topic berikutnya secara atomic.
 
    Ini mengganti pola lama:
        process = check_increment_process(...)   # baca
        ... query arxiv yang lambat ...
        set_increment_process(...)               # tulis, lama setelah baca
 
    yang rawan race condition kalau >1 worker jalan bersamaan: dua worker
    bisa baca pointer yang sama sebelum salah satu menulis ulang, sehingga
    topic yang sama diproses dobel dan salah satu update ke pointer hilang
    (lost update).
 
    Di sini, klaim + advance pointer terjadi dalam critical section yang
    SANGAT PENDEK (cuma baca+tulis redis). Lock dilepas SEBELUM task lanjut
    query ke arxiv API yang lambat, jadi worker lain tidak perlu ikut
    menunggu network call.
    """
    if not serving_topics:
        raise ValueError(
            f"serving_topics tidak boleh kosong (repository={repository})"
        )
 
    r = _get_redis()
    lock_key = f"lock:increment:{repository}"
 
    # timeout=10 -> auto-release kalau proses crash sebelum sempat unlock,
    # supaya tidak ada deadlock permanen. blocking_timeout=5 -> kalau lock
    # dipegang worker lain lebih dari 5 detik, lempar error (jangan nunggu
    # selamanya di sini karena bagian ini seharusnya sangat cepat).
    with r.lock(lock_key, timeout=10, blocking_timeout=5):
        process = check_increment_process(repository=repository)
        topic = process.get("topic", "") if process else ""
 
        # safety net: topic dari redis sudah tidak valid lagi (misal
        # serving_topics berubah setelah deploy baru) -> reset ke topic
        # pertama. serving_topics sudah divalidasi non-empty di atas jadi
        # aman untuk index [0].
        if topic not in serving_topics:
            topic = serving_topics[0]
 
        topic_index = serving_topics.index(topic)
        next_index = (topic_index + 1) % len(serving_topics)
        next_topic = serving_topics[next_index]
 
        # majukan pointer SEKARANG, sebelum query arxiv dijalankan, supaya
        # worker lain yang datang bersamaan langsung lihat pointer baru.
        # "start" di sini cuma informational (start yang otoritatif tetap
        # disimpan per-topic lewat set_topic_start/get_topic_start), jadi
        # placeholder 0 di sini aman.
        set_increment_process(repository=repository, topic=next_topic, start=0)
 
    return {"topic": topic, "next_topic": next_topic, "process": process}
 
 
def get_topic_start(repository: str, topic: str) -> int:
    """Ambil offset paging untuk topic ini. Default 0 kalau belum pernah diproses."""
    r = _get_redis()
    # NOTE: sama seperti di check_increment_process -- cast() di sini cuma
    # menyingkirkan cabang Awaitable dari stub redis-py, bukan mengklaim
    # data sudah di-decode. Decode isi tetap lewat _decode() di bawah.
    raw = cast(Optional[bytes], r.hget(f"scrape_topic_start:{repository}", topic))
    value = _decode(raw) if raw is not None else None
 
    if value is None:
        # coba ambil dari db
        try:
            depot = ScrapeProgressDepot(db_pool)
            value = str(depot.get_start_offset(repository, topic, 0))
        except Exception as e:
            log.error(
                "dedup.failed_to_get_topic_start_from_db",
                repository=repository,
                topic=topic,
                error=str(e),
                exc_info=True,
            )
            # PERHATIAN: fallback ke 0 di sini berarti kalau redis kehilangan
            # key (restart/eviction) DAN db read error bersamaan, topic ini
            # akan mulai scrape ulang dari halaman 0. Dampaknya dibatasi oleh
            # is_already_processed() di caller (tidak duplikat ingest), tapi
            # tetap buang-buang quota API. Kalau ini krusial, ganti jadi
            # `raise` supaya task di-retry oleh Celery daripada diam-diam
            # lanjut dengan start=0.
            return 0
 
    return int(value) if value is not None else 0
 
 
def set_topic_start(repository: str, topic: str, start: int) -> None:
    r = _get_redis()
    r.hset(f"scrape_topic_start:{repository}", topic, str(start))
 
    # set di db juga
    try:
        depot = ScrapeProgressDepot(db_pool)
        depot.set_progress(repository, topic, start)
    except Exception as e:
        log.error(
            "dedup.failed_to_set_topic_start_in_db",
            repository=repository,
            topic=topic,
            start=start,
            error=str(e),
            exc_info=True,
        )
        # PERHATIAN: redis berhasil ditulis, db gagal -> kedua store sekarang
        # divergen tanpa alert lain selain log ini. Kalau redis pernah hilang
        # datanya, get_topic_start() di atas akan fallback ke nilai db yang
        # sudah basi. Pertimbangkan retry/alerting kalau konsistensi db itu
        # penting buat sistem lain yang membaca ScrapeProgressDepot langsung.
        return
 
 
def reset_topic_start(repository: str, topic: str) -> None:
    r = _get_redis()
    r.hdel(f"scrape_topic_start:{repository}", topic)
 
    # reset di db juga
    try:
        depot = ScrapeProgressDepot(db_pool)
        depot.set_progress(repository, topic, 0)
    except Exception as e:
        log.error(
            "dedup.failed_to_reset_topic_start_in_db",
            repository=repository,
            topic=topic,
            error=str(e),
            exc_info=True,
        )
        return