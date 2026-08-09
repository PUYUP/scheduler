import subprocess
import json
import tempfile
import os
import structlog
import httpx

from pathlib import Path
from typing import Dict, Any
from atlazer.ingestion.providers.base import BaseRepositoryProvider
from atlazer.config.settings import settings
from atlazer.ingestion.schemas import PaperMetadata
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

CURRENT_DIR = Path(__file__).resolve().parent.parent   # .../src/atlazer/ingestion
ATLAZER_DIR = CURRENT_DIR.parent                       # .../src/atlazer
SCRAPY_DIR  = ATLAZER_DIR / "scrapy_app"               # .../src/atlazer/scrapy_app

log = structlog.get_logger(__name__)


class IaeScoreProvider(BaseRepositoryProvider):
    provider_name: str = "iaescore"
    journal: str
    issue_number: str
    article_number: str

    def __init__(self):
        self.limit = settings.max_results_per_topic

    def can_handle(self, target: str) -> bool:
        return True

    def fetch_page(
        self, 
        journal: str | None = None,
        issue_number: str | None = None,
        article_number: str | None = None,
        page: int | None = None,
    ) -> Dict[str, Any]:
        self.journal = journal if journal is not None else ""
        self.issue_number = issue_number if issue_number is not None else ""
        self.article_number = article_number if article_number is not None else ""
        self.page = page if page is not None else 1

        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        log.info(
            "Scrapy start",
            journal=self.journal,
            issue_number=self.issue_number,
            article_number=self.article_number,
            page=self.page,
        )

        try:
            result = subprocess.run(
                [
                    "scrapy", "crawl", self.provider_name,
                    "-a", f"journal={self.journal}",
                    "-a", f"issue_number={self.issue_number}",
                    "-a", f"article_number={self.article_number}",
                    "-a", f"page={self.page}",
                    "-o", output_path,
                    "-s", "LOG_LEVEL=INFO",
                ],
                cwd=str(SCRAPY_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            log.error(
                "Scrapy timeout setelah 300s. stdout=%s stderr=%s",
                e.stdout, e.stderr,
            )
            os.unlink(output_path)
            raise RuntimeError("Scrapy crawl timeout") from e

        if result.returncode != 0:
            os.unlink(output_path)
            log.error("Scrapy stderr: %s", result.stderr)
            raise RuntimeError(f"Scrapy gagal (exit {result.returncode}): {result.stderr}")

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                items = json.loads(content) if content else []
        finally:
            os.unlink(output_path)

        log.info("Scrapy done", count=len(items))

        return {"items": items}

    def extract_paper(
        self,
        url: str | None = None,
        paper_id: str | None = None,
        paper: Dict[str, Any] | None = None,
    ) -> PaperMetadata:
        if paper is None:
            raise ValueError(f"No paper provided for repository={self.provider_name}")

        paper_metadata = paper.get("paper", {})
        log.info(f"[{self.provider_name}] extract_metadata.paper", paper=paper_metadata)

        return PaperMetadata(
            paper_id=str(paper_metadata.get("article_number")),
            title=paper_metadata.get("article_title"),
            abstract=paper_metadata.get("abstract"),
            authors=paper_metadata.get("authors", []),
            categories=paper_metadata.get("keywords", []),
            source_url=paper_metadata.get("url"),
            pdf_url=paper_metadata.get("pdf_url"),
            download_url=paper_metadata.get("download_url"),
            repository=self.provider_name,
            journal=self.journal,
            doi=paper_metadata.get("doi"),
            attributes={
                "next_page": paper.get("next_page", 1),
                "next_issue_number": paper.get("next_issue_number", 1),
                "next_article_number": paper.get("next_article_number"),
            }
        )

    def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = metadata.get("paper_id")
        download_url = metadata.get("download_url")
        journal = metadata.get("journal")
        repository = metadata.get("repository")

        log.info(
            f"{repository}.download_pdf.start", 
            paper_id=paper_id, 
            journal=journal,
            url=download_url
        )

        if not paper_id:
            raise ValueError(f"No paper_id found for repository={self.provider_name}")

        if not download_url:
            raise ValueError(f"No download URL found for paper_id={paper_id} repository={self.provider_name}")

        if not repository or not journal:
            raise ValueError(f"No repository or journal found for paper_id={paper_id} repository={self.provider_name}")

        dest_path = Path(settings.pdf_download_dir) / repository / journal / f"{paper_id}.pdf"
        if dest_path.exists():
            log.info(f"{repository}.download_pdf.cache_hit", paper_id=paper_id)
            metadata["local_pdf_path"] = str(dest_path)
            return metadata

        try:
            self._download_file(download_url, dest_path)
            log.info(f"{repository}.download_pdf.success", paper_id=paper_id)
        except Exception as exc:
            log.error(f"{repository}.download_pdf.failed", paper_id=paper_id, error=str(exc))
            raise ValueError(f"Download PDF failed for paper_id={paper_id} repository={self.provider_name}")

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.pdf_max_size_mb:
            dest_path.unlink(missing_ok=True)
            log.warning(
                f"{self.provider_name}.download_pdf.too_large",
                paper_id=paper_id,
                size_mb=round(size_mb, 1),
            )
            # Don't retry — just skip this paper
            metadata["local_pdf_path"] = None
            metadata["skip_reason"] = f"PDF too large ({size_mb:.1f} MB)"
            return metadata

        log.info(
            f"{self.provider_name}.download_pdf.done",
            paper_id=paper_id,
            size_mb=round(size_mb, 2),
        )
        metadata["local_pdf_path"] = str(dest_path)
        return metadata

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=120),
    )
    def _download_file(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream(
            "GET",
            url,
            timeout=settings.download_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "atlanize-rag-ingestion/1.0 (research purposes)"},
        ) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
