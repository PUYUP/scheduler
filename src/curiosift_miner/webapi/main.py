from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from curiosift_miner.webapi.config import settings
from curiosift_miner.webapi.embeddings import EmbeddingService
from curiosift_miner.webapi.rag import RAGService
from curiosift_miner.webapi.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from curiosift_miner.webapi.vector_store import VectorStore

rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    # Model di-load SEKALI saat startup (bukan per-request) supaya
    # latency tiap request rendah. Model diambil dari HF_HOME yang
    # sudah di-mount dari folder .cache di host.
    embedding_service = EmbeddingService(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.HF_HOME,
        device=settings.DEVICE,
    )
    vector_store = VectorStore(
        dim=embedding_service.dimension,
        index_path=settings.VECTOR_STORE_PATH,
    )
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    yield

    vector_store.save()


app = FastAPI(
    title="RAG Service (Sentence-Transformers + BGE-M3)",
    description="Endpoint untuk ingest dokumen dan retrieval berbasis embedding BGE-M3.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service belum siap, model masih loading.")
    return rag_service


@app.get("/health", response_model=HealthResponse)
def health():
    service = rag_service
    return HealthResponse(
        status="ok" if service is not None else "loading",
        model=settings.MODEL_NAME,
        total_documents=service.vector_store.total_documents if service else 0,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    service = _get_service()
    chunks_added = service.ingest_documents(payload.documents)
    return IngestResponse(
        chunks_added=chunks_added,
        total_documents=service.vector_store.total_documents,
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    service = _get_service()
    results = service.query(payload.query, top_k=payload.top_k)
    return QueryResponse(query=payload.query, results=results)


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest):
    """Utility endpoint: ambil raw embedding tanpa lewat vector store."""
    service = _get_service()
    vectors = service.embedding_service.encode(payload.texts, is_query=payload.is_query)
    return EmbedResponse(embeddings=vectors.tolist(), dimension=vectors.shape[1])
