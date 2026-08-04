import arxiv
import requests
import structlog
import httpx

from pathlib import Path
from typing import List, Dict, Any
from atlazer.ingestion.providers.base import BaseRepositoryProvider
from atlazer.ingestion.schemas import PaperMetadata
from atlazer.config.settings import settings
from atlazer.utils.dedup import (
    claim_next_topic, 
    get_topic_start, 
    set_topic_start,
    is_already_processed,
    mark_as_queued,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)


class TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", 30.0)
        return super().request(*args, **kwargs)


class ArxivProvider(BaseRepositoryProvider):
    provider_name: str = "arxiv"

    def __init__(self):
        self.limit = settings.max_results_per_topic

    def can_handle(self, target: str) -> bool:
        return "arxiv.org" in target

    def fetch_page(self) -> Dict[str, Any]:
        log.info(
            f"{self.provider_name}.fetch_page.start", 
            offset=self.offset, 
            sort_by=self.sort_by
        )
        
        # construct topics to be used
        topics = settings.arxiv_topics
        topics = list(topics) if topics else []
        topics = list(dict.fromkeys(topics))

        # build next topic
        claim = claim_next_topic(repository=self.provider_name, serving_topics=topics)
        topic = claim["topic"]
        next_topic = claim["next_topic"]
        process = claim["process"]
        papers: List[Dict[str, Any]] = []

        start = get_topic_start(repository=self.provider_name, topic=topic)
 
        log.info(
            f"{self.provider_name}.fetch_page.start",
            topic=topic,
            start=start,
            max_results=self.limit,
            process=process,
        )

        try:
            results = list(self._query_arxiv(
                topic,
                sort_by=self.sort_by,
                max_results=self.limit + start,
                start=start,
            ))
        except Exception as exc:
            # FIX #3: tambahkan exc_info supaya stacktrace asli tetap tercatat,
            # tidak cuma str(exc). except Exception tetap luas karena client
            # arxiv bisa lempar bermacam-macam error (network, parsing, dll),
            # tapi minimal traceback lengkap ada di log untuk debugging.
            log.error(
                f"{self.provider_name}.fetch_page.query_failed",
                topic=topic,
                start=start,
                error=str(exc),
                exc_info=True,
            )
            raise ValueError(str(exc))
    
        log.info(
            f"{self.provider_name}.fetch_page.results_count",
            topic=topic,
            start=start,
            # FIX #5: jangan log objek `results` mentah (bisa besar / berisi
            # data yang tidak perlu masuk log). Cukup id-nya saja.
            result_ids=[r.entry_id.split("/")[-1] for r in results],
            count=len(results),
        )
    
        # tidak ada hasil lagi -> topic ini selesai untuk sekarang, reset start-nya sendiri.
        # Pointer round-robin (topic -> next_topic) sudah diklaim & dipindah secara
        # atomic di claim_next_topic() di atas, jadi tidak perlu set_increment_process
        # lagi di sini.
        if len(results) <= 0:
            # ArXiv API dikenal kadang balas feed kosong dengan status sukses
            # meskipun sebenarnya ada data (lihat: github.com/lukasschwab/arxiv.py/issues/43, #129).
            # arxiv.py versi terbaru MENERIMA ini sebagai "tidak ada hasil" tanpa
            # exception, jadi kita tidak bisa mengandalkan try/except untuk
            # mendeteksinya -- harus dicurigai manual, terutama di start=0.
            if start == 0:
                log.warning(
                    f"{self.provider_name}.fetch_page.suspicious_empty_first_page", 
                    topic=topic
                )
                # percobaan berikutnya dimulai dari 1
                set_topic_start(repository=self.provider_name, topic=topic, start=1)

            log.info(
                f"{self.provider_name}.fetch_page.next_topic",
                topic=next_topic,
                process=process,
            )

            return {
                "offset": 0,
                "topic": next_topic,
                "repository": self.provider_name,
                "process": process,
                "papers": papers,
            }

        for result in results:
            paper_id = result.entry_id.split("/")[-1]
            if is_already_processed(paper_id, repository=self.provider_name):
                continue
            mark_as_queued(paper_id, repository=self.provider_name)
            papers.append({'id': paper_id, 'pdf_url': None})
    
        # next_start ini MILIK topic yang barusan diproses, bukan untuk next_topic
        next_start = start + len(results)
        set_topic_start(repository=self.provider_name, topic=topic, start=next_start)

        return {
            "offset": next_start,
            "topic": next_topic,
            "repository": self.provider_name,
            "process": process,
            "papers": papers,
        }

    def extract_paper(self, url: str | None, paper_id: str) -> PaperMetadata:
        log.info(
            f"{self.provider_name}.extract_paper.start", 
            paper_id=paper_id
        )

        try:
            paper = self._fetch_single_paper(paper_id)
        except Exception as exc:
            log.warning("scrape_paper_metadata.fetch_failed", paper_id=paper_id, repository=self.provider_name, error=str(exc))
            raise ValueError(f"No paper found for id={paper_id} repository={self.provider_name}")

        log.info(
            f"{self.provider_name}.extract_paper.results", 
            paper_id=paper_id, 
            paper=paper
        )

        return PaperMetadata(
            paper_id=paper_id,
            repository=self.provider_name,
            title=paper.title.strip().replace("\n", " "),
            abstract=paper.summary.strip().replace("\n", " "),
            authors=[a.name for a in paper.authors],
            categories=paper.categories,
            published=paper.published.isoformat(),
            updated=paper.updated.isoformat(),
            pdf_url=paper.pdf_url,
            doi=paper.doi or "",
            journal_ref=paper.journal_ref or "",
            primary_category=paper.primary_category if paper.primary_category else "",
            metadata={"doi": "10.1000/182"}
        )

    def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = metadata.get("paper_id")
        pdf_url = metadata.get("pdf_url")

        log.info(
            f"{self.provider_name}.download_pdf.start", 
            paper_id=paper_id, 
            url=pdf_url
        )

        if not paper_id:
            raise ValueError(f"No paper_id found for repository={self.provider_name}")

        if not pdf_url:
            raise ValueError(f"No PDF URL found for paper_id={paper_id} repository={self.provider_name}")

        dest_path = Path(settings.pdf_download_dir) / self.provider_name / paper_id / f"{paper_id}.pdf"
        if dest_path.exists():
            log.info(f"{self.provider_name}.download_pdf.cache_hit", paper_id=paper_id)
            metadata["local_pdf_path"] = str(dest_path)
            return metadata

        try:
            self._download_file(pdf_url, dest_path)
            log.info(f"{self.provider_name}.download_pdf.success", paper_id=paper_id)
        except Exception as exc:
            log.error(f"{self.provider_name}.download_pdf.failed", paper_id=paper_id, error=str(exc))
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

    def _fetch_single_paper(self, paper_id: str):
        client = arxiv.Client(num_retries=3, delay_seconds=5)
        client._session = TimeoutSession()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        if not results:
            raise ValueError(f"No paper found for id={paper_id} provider={self.provider_name}")
        return results[0]

    def _query_arxiv(
        self,
        topic: str, 
        max_results: int, 
        sort_by: str, 
        start: int = 0
    ):
        log.info(
            f"{self.provider_name}.query_arxiv.start", 
            topic=topic, 
            max_results=max_results, 
            sort_by=sort_by, 
            start=start
        )

        sort_criterion = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "relevance":     arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        client = arxiv.Client(
            page_size=min(max_results, 100),
            delay_seconds=10,          # respect ArXiv rate limit
            num_retries=5,
        )
        client._session = TimeoutSession()
        search = arxiv.Search(
            query=f"cat:{topic}",
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )
        return list(client.results(search, offset=start))

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
    )
    def _download_file(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream(
            "GET",
            url,
            timeout=settings.download_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "atlanize-rag-scraper/1.0 (research purposes)"},
        ) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)