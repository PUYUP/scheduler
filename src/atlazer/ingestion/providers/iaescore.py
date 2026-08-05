import subprocess

from pathlib import Path
from typing import Dict, Any
from atlazer.ingestion.providers.base import BaseRepositoryProvider
from atlazer.config.settings import settings
from atlazer.ingestion.schemas import PaperMetadata

CURRENT_DIR = Path(__file__).resolve().parent.parent   # .../src/atlazer/ingestion
ATLAZER_DIR = CURRENT_DIR.parent                       # .../src/atlazer
SCRAPY_DIR  = ATLAZER_DIR / "scrapy_app"               # .../src/atlazer/scrapy_app


class IaeScoreProvider(BaseRepositoryProvider):
    provider_name: str = "iaescore"
    journal_name: str = "ijai"
    volume: str = "588"

    def __init__(self):
        self.limit = settings.max_results_per_topic

    def can_handle(self, target: str) -> bool:
        return True

    def fetch_page(self) -> Dict[str, Any]:
        result = subprocess.run(
            [
                "scrapy", "crawl", self.provider_name,
                "-a", f"journal={self.journal_name}",
                "-a", f"volume={self.volume}"
            ],  
            cwd=str(SCRAPY_DIR),
            capture_output=True,
            text=True
        )

        return {}

    def extract_paper(self, url: str | None, paper_id: str) -> PaperMetadata:
        return PaperMetadata(
            paper_id=paper_id,
            title="test",
            abstract="test",
            source_url="test",
            download_url="test",
            created_at="2022-01-01T00:00:00Z",
            repository=self.provider_name,
            journal=self.journal_name,
        )

    def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {}