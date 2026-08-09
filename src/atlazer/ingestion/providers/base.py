from abc import ABC, abstractmethod
from typing import Dict, Any
from atlazer.ingestion.schemas import PaperMetadata


class BaseRepositoryProvider(ABC):
    provider: str
    journal: str
    issue_number: int
    article_number: int
    limit: int = 1
    offset: int = 0
    sort_by: str = "newest"

    @abstractmethod
    def can_handle(self, target: str) -> bool:
        """URL checking matching with this provider"""
        pass

    @abstractmethod
    def fetch_page(
        self,
        journal: str | None = None,
        issue_number: str | None = None,
        article_number: str | None = None,
    ) -> Dict[str, Any]:
        """Fetch a single page of papers from the repository. Return list of paper {id, url}"""
        pass

    @abstractmethod
    def extract_paper(
        self,
        url: str | None = None,
        paper_id: str | None = None,
        paper: Dict[str, Any] | None = None
    ) -> PaperMetadata:
        """Extract base data (title, authors, abstract, etc) based on repo source."""
        pass

    @abstractmethod
    def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Download PDF from the repository."""
        pass        