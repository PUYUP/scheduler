"""
config/celery_config.py
────────────────────────
All Celery settings in one place.
Loaded via: app.config_from_object("config.celery_config")
"""

from atlazer.config.settings import settings

# ─── Task Discovery ───────────────────────────────────────────────────────────
# Listed explicitly — autodiscover with related_name="" is unreliable in Docker.
# Every module with @app.task must appear here.
imports = (
    "atlazer.celery_app.tasks.scrape",
    "atlazer.celery_app.tasks.process",
    "atlazer.celery_app.tasks.embed",
    "atlazer.celery_app.tasks.store",
    "atlazer.celery_app.tasks.webapi",
    "atlazer.celery_app.tasks.matcher",
    "atlazer.celery_app.tasks.challenge",
    "atlazer.celery_app.tasks.evaluation",
    "atlazer.celery_app.tasks.maintenance",
)

# ─── Broker & Backend ─────────────────────────────────────────────────────────
broker_url                  = settings.redis_url
result_backend              = settings.redis_url
broker_connection_retry_on_startup = True

# ─── Serialization ────────────────────────────────────────────────────────────
task_serializer             = "json"
result_serializer           = "json"
accept_content              = ["json"]
timezone                    = "UTC"
enable_utc                  = True

# ─── Result Storage ───────────────────────────────────────────────────────────
result_expires              = 86_400          # 24 h
result_compression          = "gzip"
result_extended             = True            # store task name, args, kwargs

# ─── Reliability ──────────────────────────────────────────────────────────────
worker_proc_alive_timeout   = settings.worker_proc_alive_timeout
task_acks_late              = True            # ack AFTER task completes
task_reject_on_worker_lost  = True            # re-queue if worker dies mid-task
task_track_started          = True
worker_prefetch_multiplier  = 1               # one task at a time per worker slot
                                              # (overridden per worker CLI flag)

# ─── Retry Defaults (each task can override) ──────────────────────────────────
task_annotations = {
    "atlazer.celery_app.tasks.scrape.*": {
        "max_retries": 5,
        "default_retry_delay": 60,
        "time_limit": 180,
        "soft_time_limit": 150,
    },
    "atlazer.celery_app.tasks.process.*": {
        "max_retries": 3,
        "default_retry_delay": 120,
        "time_limit": 1800,
        "soft_time_limit": 1500,
    },
    "atlazer.celery_app.tasks.embed.*": {
        "max_retries": 10,
        "default_retry_delay": 30,
        "rate_limit": "20/m",        # stay under embedding API rate limit
        "time_limit": 3600,
        "soft_time_limit": 3300,
    },
    "atlazer.celery_app.tasks.store.*": {
        "max_retries": 10,
        "default_retry_delay": 30,
        "time_limit": 1800,
        "soft_time_limit": 1700,
    },
    "atlazer.celery_app.tasks.webapi.*": {
        "max_retries": 3,
        "default_retry_delay": 60,
        "time_limit": 3600,
        "soft_time_limit": 3300,
    },
    "atlazer.celery_app.tasks.matcher.*": {
        "max_retries": 3,
        "default_retry_delay": 60,
        "time_limit": 3600,
        "soft_time_limit": 3300,
    },
    "atlazer.celery_app.tasks.challenge.*": {
        "max_retries": 3,
        "default_retry_delay": 60,
        "time_limit": 3600,
        "soft_time_limit": 3300,
    },
    "atlazer.celery_app.tasks.evaluation.*": {
        "max_retries": 3,
        "default_retry_delay": 60,
        "time_limit": 3600,
        "soft_time_limit": 3300,
    },
}

# ─── Task Routing ─────────────────────────────────────────────────────────────
task_routes = {
    # ── Scrape tier ──
    "atlazer.celery_app.tasks.scrape.scrape_topic":             {"queue": "scrape"},
    "atlazer.celery_app.tasks.scrape.scrape_topic_backfill":    {"queue": "scrape"},
    "atlazer.celery_app.tasks.scrape.scrape_topic_increment":   {"queue": "scrape"},
    "atlazer.celery_app.tasks.scrape.scrape_paper_metadata":    {"queue": "scrape"},
    "atlazer.celery_app.tasks.scrape.download_pdf":             {"queue": "scrape"},

    # ── Process tier ──
    "atlazer.celery_app.tasks.process.parse_pdf":            {"queue": "process"},
    "atlazer.celery_app.tasks.process.clean_text":           {"queue": "process"},
    "atlazer.celery_app.tasks.process.chunk_document":       {"queue": "process"},

    # ── Embed tier ──
    "atlazer.celery_app.tasks.embed.generate_embeddings":    {"queue": "embed"},

    # ── Store tier ──
    "atlazer.celery_app.tasks.store.store_paper":            {"queue": "store"},

    # ── WebAPI tier ──
    "atlazer.celery_app.tasks.webapi.generate_embeddings":   {"queue": "webapi"},

    # ── Matcher tier ──
    "atlazer.celery_app.tasks.matcher.single_user":          {"queue": "matcher"},
    "atlazer.celery_app.tasks.matcher.batch_user":           {"queue": "matcher"},
    "atlazer.celery_app.tasks.matcher.summarize_paper":      {"queue": "matcher"},

    # ── Challenge tier ──
    "atlazer.celery_app.tasks.challenge.chunk_answer":              {"queue": "challenge"},
    "atlazer.celery_app.tasks.challenge.embed_answer":              {"queue": "challenge"},
    "atlazer.celery_app.tasks.challenge.save_embedding_answer":     {"queue": "challenge"},
    "atlazer.celery_app.tasks.challenge.process_challenge_papers":  {"queue": "challenge"},
    "atlazer.celery_app.tasks.challenge.process_answer_similarity": {"queue": "challenge"},
    "atlazer.celery_app.tasks.challenge.save_answer_similarity":    {"queue": "challenge"},

    # ── Evaluation tier ──
    "atlazer.celery_app.tasks.evaluation.generate_jsonl":           {"queue": "evaluation"},
    "atlazer.celery_app.tasks.evaluation.scoring_answer":           {"queue": "evaluation"},

    # ── Maintenance ──
    "atlazer.celery_app.tasks.maintenance.*":   {"queue": "default"},
}

# ─── Worker ───────────────────────────────────────────────────────────────────
worker_send_task_events     = True            # enables Flower real-time updates
task_send_sent_event        = True
worker_hijack_root_logger   = False           # use our structlog config

# ─── Chord / Group behaviour ──────────────────────────────────────────────────
# When a chord header fails, don't run the callback
chord_propagates_errors     = True

# ─── Broker Transport Options ─────────────────────────────────────────────────
broker_transport_options = {
    "visibility_timeout": 7200,      # 2 h — must be ≥ longest task
    "retry_policy": {
        "max_retries": 3,
    },
}