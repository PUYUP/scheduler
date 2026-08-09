from __future__ import annotations
from typing import Any, Dict, List

import structlog

from celery import group, signature
from atlazer.celery_app.main import app
from atlazer.ingestion.providers.iaescore import IaeScoreProvider
from atlazer.utils.dedup import (
    get_ingestion_process,
    set_ingestion_process,
    get_current_issue,
    get_current_issue_status,
    get_current_page,
    get_current_paper,
)

log = structlog.get_logger(__name__)
provider = IaeScoreProvider()
repository = provider.provider_name
QUEUE = "iaescore"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 5 — fetch_page
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.iaescore.fetch_page",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    time_limit=600,
    soft_time_limit=540,
    queue=QUEUE,
    ignore_result=False,
)
def fetch_page(self, journal: str) -> Dict[str, Any]:
    log.info("iaescore.fetch_page.start")

    current_page = get_current_page(journal=journal, repository="iaescore")
    current_issue = get_current_issue(journal=journal, repository="iaescore")
    current_paper = get_current_paper(journal=journal, repository="iaescore")

    if current_issue:
        current_issue_status = get_current_issue_status(
            journal=journal,
            repository="iaescore",
            issue_number=current_issue
        )

        log.info(
            "iaescore.fetch_page.current_issue_status",
            current_issue_status=current_issue_status
        )
    else:
        current_issue_status = None

    result = provider.fetch_page(
        journal=journal,
        page=int(current_page) if current_page else 1,
        issue_number=current_issue if current_issue else "",
        article_number=current_paper if current_paper else "",
    )

    log.info("iaescore.fetch_page.result", result=result)

    if not result:
        log.warning("Tidak ada item baru dari IaeScore")
        return {}

    papers = result.get("items", [])
    # job = group(extract_metadata.s(paper=p).set(queue=QUEUE) for p in papers)
    # job.apply_async()

    # # add job id to metadata
    # result["job_id"] = job.id

    for p in papers:
        extract_metadata(paper=p)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 5 — extract_metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.iaescore.extract_metadata",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    time_limit=600,
    soft_time_limit=540,
    queue=QUEUE,
    ignore_result=False,
)
def extract_metadata(
    self,
    paper: Dict[str, Any],
) -> Dict[str, Any]:
    log.info(f"[{repository}] extract_metadata.start", paper=paper)

    try:
        metadata = provider.extract_paper(paper=paper)
    except Exception as err:
        log.error(f"[{repository}] extract_metadata failed: {str(err)}")
        self.retry(exc=err)

    # convert to dict
    metadata_dict = metadata.model_dump(exclude_none=True)

    log.info(f"[{repository}] extract_metadata.result", metadata=metadata_dict)

    # process to next tasks
    # (
    #     download_pdf.s(metadata=metadata_dict).set(queue=QUEUE)
    #     | signature(
    #         "atlazer.celery_app.tasks.ingestion.process.parse_pdf",
    #         queue="process",
    #         immutable=False,
    #     )
    # ).apply_async()

    download_pdf(metadata=metadata_dict)

    log.info(
        f"[{repository}] extract_metadata.result",
        metadata=metadata_dict
    )

    return metadata_dict


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 5 — download_pdf
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.ingestion.extractors.iaescore.download_pdf",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    time_limit=600,
    soft_time_limit=540,
    queue=QUEUE,
    ignore_result=False,
)
def download_pdf(
    self,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    log.info(f"[{repository}] download_pdf.start", metadata=metadata)

    try:
        metadata = provider.download_pdf(metadata)
    except Exception as err:
        log.error(f"[{repository}] download_pdf failed: {str(err)}")
        self.retry(exc=err)

    if metadata:
        attributes = metadata.get("attributes", {})
        ingest_data = set_ingestion_process(
            repository=metadata.get("repository", ""),
            journal=metadata.get("journal", ""),
            page=int(attributes.get("next_page", 1)),
            issue_number=int(attributes.get("next_issue_number", 0)),
            paper_id=metadata.get("paper_id", ""),
        )

        log.info(f'[{repository}] ingestion_process_setter', ingest_data=ingest_data)

    return metadata
