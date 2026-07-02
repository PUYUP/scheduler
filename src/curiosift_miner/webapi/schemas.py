from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    documents: List[DocumentIn]


class IngestResponse(BaseModel):
    chunks_added: int
    total_documents: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model: str
    total_documents: int
