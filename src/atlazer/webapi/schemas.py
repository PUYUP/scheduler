from typing import Any, Dict, List
from pydantic import BaseModel


class EmbedChunksRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    provision: Dict[str, Any] | None = None


class EmbedChunksResponse(BaseModel):
    chunks: List[Dict[str, Any]]


class EmbedParallelResponse(BaseModel):
    task_id: str


class HealthResponse(BaseModel):
    status: str
    model: str
