from typing import Any, Dict, List
from pydantic import BaseModel

class EmbedChunksRequest(BaseModel):
    chunks: List[Dict[str, Any]]

class EmbedChunksResponse(BaseModel):
    chunks: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model: str
