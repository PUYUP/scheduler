from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from atlazer.ingestion.schemas import PaperMetadata


class BaseRepositoryProvider(ABC):
    provider_name: str
    limit: int = 1
    offset: int = 0
    sort_by: str = "newest"

    @abstractmethod
    def can_handle(self, target: str) -> bool:
        """URL checking matching with this provider"""
        pass

    @abstractmethod
    def fetch_page(self) -> Dict[str, Any]:
        """Fetch a single page of papers from the repository. Return list of paper {id, url}"""
        pass

    @abstractmethod
    def extract_paper(self, url: str | None, paper_id: str) -> PaperMetadata:
        """Extract base data (title, authors, abstract, etc) based on repo source."""
        pass

    @abstractmethod
    def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Download PDF from the repository."""
        pass        