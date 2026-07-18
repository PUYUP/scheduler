from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TaskExecutionResponse(BaseModel):
    task_id: str


class EmbedChunksRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    provision: Dict[str, Any] | None = None


class EmbedAnswerRequest(BaseModel):
    user_id: str
    challenge_id: str
    answer_id: str
    content: str
    language_code: str


class PaperMatcherRequest(BaseModel):
    user_id: str
    language_code: str = 'en'


class EmbedChunksResponse(BaseModel):
    chunks: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model: str


class EvaluateAnswerRequest(BaseModel):
    user_id: str
    challenge_id: str
    answer_id: str
