from typing import List

from app.embeddings import EmbeddingService
from app.schemas import DocumentIn, RetrievedChunk
from app.vector_store import VectorStore


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunking sederhana berbasis karakter dengan overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size harus lebih besar dari chunk_overlap")

    text = text.strip()
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_documents(self, documents: List[DocumentIn]) -> int:
        all_chunks: List[str] = []
        all_metadatas: List[dict] = []

        for doc in documents:
            pieces = chunk_text(doc.text, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(pieces)
            all_metadatas.extend([doc.metadata or {} for _ in pieces])

        if not all_chunks:
            return 0

        embeddings = self.embedding_service.encode(all_chunks, is_query=False)
        self.vector_store.add(embeddings, all_chunks, all_metadatas)
        return len(all_chunks)

    def query(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_embedding = self.embedding_service.encode([query], is_query=True)
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        return [RetrievedChunk(**r) for r in raw_results]
