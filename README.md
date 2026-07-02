# ArXiv RAG Scraping Pipeline

Celery + Redis pipeline that ingests ArXiv papers end-to-end:
**discover → download → parse → chunk → embed → store**.

---

## Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │              Docker Compose                      │
                         │                                                  │
  ┌──────────┐  every   │  ┌──────────┐  ┌──────────────────────────────┐ │
  │   Beat   │──6 hrs──►│  │  SCRAPE  │  │  Queue Topology              │ │
  │Scheduler │          │  │ workers  │  │                              │ │
  └──────────┘          │  │ (×2)     │  │  scrape ──DLX──► dlx.scrape │ │
                         │  └────┬─────┘  │  process──DLX──►dlx.process │ │
                         │       │ chain  │  embed ──DLX──► dlx.embed   │ │
                         │  ┌────▼─────┐  └──────────────────────────────┘ │
                         │  │ PROCESS  │                                  │
                         │  │ workers  │   Beat also runs every hour:    │
                         │  │ (×2)     │   • retry_dead_letters (×3)     │
                         │  └────┬─────┘   • pipeline_health (×15min)   │
                         │       │ chain   • purge_old_pdfs (×daily)     │
                         │  ┌────▼─────┐                                  │
                         │  │  EMBED   │                                  │
                         │  │ workers  │                                  │
                         │  │ (×1)     │                                  │
                         │  └────┬─────┘                                  │
                         │       │                                        │
                         │  ┌────▼──────────┐  ┌────────┐               │
                         │  │ Vector Store  │  │ Flower │:5555           │
                         │  │ (your DB)     │  │Monitor │               │
                         │  └───────────────┘  └────────┘               │
                         └─────────────────────────────────────────────────┘
```

## Task Chain (7 tasks across 3 queues)

```
[SCRAPE queue]                [PROCESS queue]              [EMBED queue]
      │                              │                           │
scrape_topic                         │                           │
  │  Queries ArXiv API               │                           │
  │  Filters dedup                   │                           │
  └─► scrape_paper_metadata          │                           │
        │  Fetches full metadata     │                           │
        └─► download_pdf             │                           │
              │  HTTP stream PDF     │                           │
              └────────────────► parse_pdf                       │
                                   │  PyMuPDF section extract    │
                                   └─► clean_text                │
                                         │  Strip boilerplate    │
                                         └─► chunk_document      │
                                               │  512-tok chunks  │
                                               └─────────────► generate_embeddings
                                                                 │  Batch embed API
                                                                 └─► store_chunks
                                                                       │  Write vectors
                                                                       └─► mark_as_processed
```

## Quick Start

```bash
# 1. Copy env template
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or switch to EMBEDDING_PROVIDER=local)

# 2. Start everything
docker compose up -d

# 3. Watch tasks in Flower
open http://localhost:5555/flower

# 4. Manually trigger a scrape
docker compose exec worker-scrape \
    python src/atlazer/scripts/trigger_scrape.py --arxiv-topics cs.CL --max-results 20

# 5. Re-ingest a specific paper
docker compose exec worker-scrape \
    python src/atlazer/scripts/trigger_scrape.py --paper-id 2606.27414 --repository arxiv
```

## Scaling

```bash
# Add more scrape workers when queue depth is high
docker compose up -d --scale worker-scrape=4

# Add more process workers for PDF-heavy workloads
docker compose up -d --scale worker-process=4

# Embed workers stay at 1–2 to respect API rate limits
docker compose up -d --scale worker-embed=2
```

## File Layout

```
arxiv-rag/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
│
├── config/
│   ├── settings.py          # Pydantic settings (reads .env)
│   ├── celery_config.py     # All Celery knobs
│   └── logging.py           # structlog JSON / console
│
├── celery_app/
│   ├── main.py              # App factory, queue topology, Beat schedule
│   ├── tasks/
│   │   ├── scrape.py        # Tasks 1–3  (scrape queue)
│   │   ├── process.py       # Tasks 4–5  (process queue)
│   │   ├── embed.py         # Tasks 6–7  (embed queue)
│   │   └── maintenance.py   # DLQ retry, health check, pdf purge
│   └── utils/
│       ├── dedup.py         # Redis-backed paper deduplication
│       ├── embedder.py      # OpenAI / local embedding abstraction
│       ├── paper_schema.py  # Pydantic models for inter-task data
│       └── text_cleaner.py  # Academic PDF text normalisation
│
└── scripts/
    └── trigger_scrape.py    # CLI to manually kick off ingestion
```

## Connecting Your Vector Store

Open `celery_app/tasks/embed.py` and replace the `_write_chunks()` stub
with your writer. Each chunk arriving there is a plain dict with:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | `<arxiv_id>_<section_idx>_<chunk_idx>` |
| `arxiv_id` | str | e.g. `2401.12345` |
| `title` | str | Paper title |
| `section` | str | Section heading |
| `text` | str | Chunk body (≈512 tokens) |
| `embedding` | list[float] | Dense vector |
| `embedding_model` | str | e.g. `text-embedding-3-small` |
| `embedding_dim` | int | 1536 (OpenAI) / 384 (local) |
| `authors` | list[str] | Author names |
| `categories` | list[str] | ArXiv category codes |
| `published` | str | ISO 8601 date |
| `token_count` | int | Approximate token count |


## To fix the "model-init exited with code 1" error

```bash
docker compose build --no-cache model-init
docker volume rm <nama-project>_hf-cache
docker compose up model-init
```

## Troubleshooting

If you see these errors after rebuilding/restarting:

```
model-init-1  | [warm_model_cache] FAILED to download ... Path ... not found
model-init-1 exited with code 1
service "model-init" didn't complete successfully
```

It means the `hf-cache` volume was created but never populated, and the
`model-init` container exited because it couldn't find the model files on its
first run. To fix:

1. **Clean the stale cache volume:**

   ```bash
   docker volume rm scheduler_hf-cache
   ```

   Replace `<nama-project>` with your Compose project name (usually the directory
   name, e.g. `scheduler` or `arxiv-rag`).

2. **Rebuild the image (no-cache) and restart:**

   ```bash
   docker compose build --no-cache model-init
   docker compose up -d
   ```

This ensures Docker rebuilds the image from scratch and `model-init` has a clean
volume to download the model into before the workers start.