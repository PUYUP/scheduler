"""
main.py
──────────────────
Celery application factory.
Import this module everywhere: `celery -A atlazer.celery_app.main ...`
"""

from celery import Celery
from celery.signals import (
    setup_logging,
    worker_ready,
    worker_process_init,
    worker_process_shutdown,
    task_failure
)
from kombu import Exchange, Queue

from atlazer.config.settings import settings
from atlazer.config.logging import configure_logging
from atlazer.storage.db import DatabasePool, DatabaseConfig
from atlazer.utils.embedder import get_embedder


# ─── Singleton ────────────────────────────────────────────────────────────────
# Dibuat DULU di module level, sebelum konfigurasi dijalankan. Ini penting
# karena _configure_beat_schedule() di bawah meng-import tasks.maintenance,
# dan modul itu sendiri melakukan `from atlazer.celery_app.main import app` — kalau
# `app` belum ter-assign di sini saat itu terjadi, akan muncul ImportError
# (circular import). Dengan app dibuat di baris pertama, atribut `app` sudah
# ada di modul ini begitu Python mulai mengeksekusi file ini.
app = Celery("atlazer")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_celery_app() -> Celery:
    app.config_from_object("atlazer.config.celery_config")
    # Explicit imports are more reliable than autodiscover across Docker
    # bind-mount layouts — every task module must be listed here.
    app.conf.include = [
        "atlazer.celery_app.tasks.scrape",
        "atlazer.celery_app.tasks.process",
        "atlazer.celery_app.tasks.embed",
        "atlazer.celery_app.tasks.maintenance",
        "atlazer.celery_app.tasks.webapi",
    ]
    _configure_queues(app)
    _configure_beat_schedule(app)
    return app


# ─── Queue Topology ───────────────────────────────────────────────────────────
# Broker = Redis. Setiap Celery queue adalah Redis list biasa (LPUSH/BRPOP).
# Redis TIDAK mendukung x-dead-letter-exchange / x-message-ttl ala AMQP —
# itu fitur RabbitMQ murni dan diam-diam diabaikan oleh Redis transport.
# Dead-lettering di sini dilakukan MANUAL lewat signal task_failure
# (lihat on_task_failure di bawah), setelah semua retry habis.
#
#  scrape  → discover papers, download PDFs
#  process → parse PDF, clean text, chunk
#  embed   → generate embeddings, store vectors
#
# Tasks chain:  scrape_topic → scrape_paper_metadata → download_pdf
#                           → parse_pdf → chunk_document
#                           → generate_embeddings → store_chunks

def _configure_queues(app: Celery) -> None:
    default_exchange = Exchange("default", type="direct")
    scrape_exchange  = Exchange("scrape",  type="direct")
    process_exchange = Exchange("process", type="direct")
    embed_exchange   = Exchange("embed",   type="direct")
    webapi_exchange  = Exchange("webapi",  type="direct")
    dlx_exchange     = Exchange("dlx",     type="direct")   # dead-letter (nama saja; tidak ada semantik khusus di Redis)

    app.conf.task_queues = (
        Queue("default", default_exchange, routing_key="default"),
        # ── Tier 1: I/O bound ──
        Queue("scrape",  scrape_exchange,  routing_key="scrape"),
        # ── Tier 2: CPU bound ──
        Queue("process", process_exchange, routing_key="process"),
        # ── Tier 3: API rate-limited ──
        Queue("embed",   embed_exchange,   routing_key="embed"),
        # ── WebAPI rate-limited ──
        Queue("webapi",  webapi_exchange,  routing_key="webapi"),
        # ── Dead-letter sinks (diisi manual via on_task_failure, dikuras via
        #     tasks.maintenance.retry_dead_letters) ──
        Queue("dlx.scrape",   dlx_exchange, routing_key="dlx.scrape"),
        Queue("dlx.process",  dlx_exchange, routing_key="dlx.process"),
        Queue("dlx.embed",    dlx_exchange, routing_key="dlx.embed"),
        Queue("dlx.webapi",   dlx_exchange, routing_key="dlx.webapi"),
    )

    app.conf.task_default_queue    = "default"
    app.conf.task_default_exchange = "default"
    app.conf.task_default_routing_key = "default"


# ─── Periodic Beat Schedule ───────────────────────────────────────────────────

def _configure_beat_schedule(app: Celery) -> None:
    """
    Periodic ingestion of ArXiv topics defined in settings, plus
    dead-letter retry and housekeeping tasks.
    """
    from atlazer.celery_app.tasks.maintenance import MAINTENANCE_BEAT_SCHEDULE

    app.conf.beat_schedule = {
        # ── Main ingestion: every 6 hours per topic ──
        **{
            f"scrape-{topic.replace(' ', '-')}-periodic": {
                "task": "atlazer.celery_app.tasks.scrape.scrape_topic",
                "schedule": settings.scrape_interval_seconds,
                "args": [topic],
                "kwargs": {"max_results": settings.max_results_per_topic},
                "options": {"queue": "scrape"},
            }
            for topic in settings.arxiv_topics
        },
        # ── Backfill trigger for each topic ──
        # Catatan: beat akan memicu ulang task ini secara periodik dari start=0.
        # Ini AMAN karena task sendiri akan skip kalau backfill sudah selesai
        # (lihat guard is_backfill_complete di scrape_topic_backfill), dan
        # berfungsi sebagai safety-net kalau chain pagination sempat terputus
        # (misal worker crash di tengah jalan).
        **{
            f"scrape-{topic.replace(' ', '-')}-backfill": {
                "task": "atlazer.celery_app.tasks.scrape.scrape_topic_backfill",
                "schedule": settings.scrape_backfill_interval_seconds,
                "args": [topic],
                "kwargs": {
                    "page_size": settings.max_results_per_topic,
                    "start": 0,
                },
                "options": {"queue": "scrape"},
            }
            for topic in settings.arxiv_topics
        },
        # ── Retry dead-letter queue items every hour ──
        "retry-failed-scrape": {
            "task": "atlazer.celery_app.tasks.maintenance.retry_dead_letters",
            "schedule": 3600,
            "args": ["dlx.scrape"],
            "options": {"queue": "default"},
        },
        "retry-failed-process": {
            "task": "atlazer.celery_app.tasks.maintenance.retry_dead_letters",
            "schedule": 3600,
            "args": ["dlx.process"],
            "options": {"queue": "default"},
        },
        "retry-failed-embed": {
            "task": "atlazer.celery_app.tasks.maintenance.retry_dead_letters",
            "schedule": 3600,
            "args": ["dlx.embed"],
            "options": {"queue": "default"},
        },
        # ── Housekeeping (purge_old_pdfs, pipeline_health) ──
        **MAINTENANCE_BEAT_SCHEDULE,
    }

    app.conf.beat_scheduler = "celery.beat.PersistentScheduler"


# ─── Signals ──────────────────────────────────────────────────────────────────

@setup_logging.connect
def on_setup_logging(**kwargs):
    configure_logging()


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    import structlog
    log = structlog.get_logger()
    log.info("worker_ready", hostname=sender.hostname)


# ─── Database Init and Shutdown ────────────────────────────────────────────────

db_pool = DatabasePool(DatabaseConfig.from_env())


@worker_process_init.connect
def init_db(**kwargs):
    db_pool.start()


@worker_process_shutdown.connect
def close_db(**kwargs):
    db_pool.close()


# ─── Preload embedder once per worker process ────────────────────────────────

@worker_process_init.connect
def preload_embedder(**kwargs):
    import structlog
    log = structlog.get_logger()
    log.info("worker_process_init.preloading_embedder")
    get_embedder()


# Nama task per-tier, dipakai untuk menentukan queue asal saat dead-lettering.
# Harus tetap sinkron dengan task_routes di config/celery_config.py.
_TIER_PREFIXES = {
    "atlazer.celery_app.tasks.scrape.":  "scrape",
    "atlazer.celery_app.tasks.process.": "process",
    "atlazer.celery_app.tasks.embed.":   "embed",
    "atlazer.celery_app.tasks.webapi.":  "webapi",
}


@task_failure.connect
def on_task_failure(sender, task_id, exception, args, kwargs, traceback, einfo, **extra):
    """
    Fires only after all task-level retries are exhausted (see
    task_annotations.max_retries per tier in celery_config.py) — i.e. this
    is a permanent failure, not a transient one.

    Note: `args`/`kwargs` here are the signal's own parameters (the failed
    task's original arguments), not read from sender.request — that context
    isn't reliably populated at failure time.
    """
    import structlog
    log = structlog.get_logger()

    task_name = sender.name
    original_queue = next(
        (q for prefix, q in _TIER_PREFIXES.items() if task_name.startswith(prefix)),
        None,
    )

    log.error(
        "task_failed",
        task_id=task_id,
        task_name=task_name,
        queue=original_queue,
        error=str(exception),
    )

    # ── Manual dead-lettering (Redis broker has no native DLX) ────────────
    # Only scrape/process/embed tasks are dead-lettered; maintenance-tier
    # failures (queue=default) are excluded to avoid a dead-letter loop.
    if original_queue is not None:
        try:
            app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                queue=f"dlx.{original_queue}",
            )
            log.info("task_dead_lettered", task_id=task_id, dlq=f"dlx.{original_queue}")
        except Exception as exc:
            log.error("dead_letter_publish_failed", task_id=task_id, error=str(exc))


# ─── Finalize ─────────────────────────────────────────────────────────────────
# app sudah dibuat di module level (baris atas); di sini kita jalankan
# konfigurasinya. create_celery_app() mengembalikan objek `app` yang sama
# (bukan instance baru), jadi ini bukan re-assignment ke objek berbeda.
app = create_celery_app()
