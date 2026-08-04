from __future__ import annotations
from typing import Any, Dict

import structlog
from celery import group, signature

from atlazer.celery_app.main import app
from atlazer.ingestion.providers.arxiv import ArxivProvider

log = structlog.get_logger(__name__)
provider = ArxivProvider()
repository = provider.provider_name


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 5 — fetch_page
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.arxiv.fetch_page",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="arxiv",
    ignore_result=False,
)
def fetch_page(self) -> Dict[str, Any]:
    try:
        result = provider.fetch_page()
    except Exception as err:
        log.error(f"[{repository}] fetch_page failed: {str(err)}")
        self.retry(exc=err)

    papers = result.get("papers", [])
    job = group(
        extract_metadata.s(
            url=p.get("pdf_url", None),
            paper_id=p.get("id", None),
        ).set(queue="arxiv") for p in papers
    )
    job.apply_async()

    # add job id to metadata
    result["job_id"] = job.id
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 5 — extract_metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.arxiv.extract_metadata",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue="arxiv",
    ignore_result=False,
)
def extract_metadata(
    self,
    url: str | None,
    paper_id: str,
) -> Dict[str, Any]:
    log.info(f"[{repository}] extract_metadata.start")

    try:
        metadata = provider.extract_paper(url, paper_id)
    except Exception as err:
        log.error(f"[{repository}] extract_metadata failed: {str(err)}")
        self.retry(exc=err)

    # convert to dict
    metadata_dict = metadata.model_dump(exclude_none=True)

    # process to next tasks
    (
        download_pdf.s(metadata=metadata_dict).set(queue="arxiv")
        | signature(
            "atlazer.celery_app.tasks.process.parse_pdf",
            queue="process",
            immutable=False,
        )
    ).apply_async()

    return metadata_dict

# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 5 — download_pdf
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.arxiv.download_pdf",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue="arxiv",
    ignore_result=False,
)
def download_pdf(
    self,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    log.info(f"[{repository}] download_pdf.start")

    try:
        metadata = provider.download_pdf(metadata)
    except Exception as err:
        log.error(f"[{repository}] download_pdf failed: {str(err)}")
        self.retry(exc=err)

    return metadata
