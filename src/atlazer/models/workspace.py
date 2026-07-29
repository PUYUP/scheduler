from uuid import UUID
from sqlalchemy import Integer, String, Index, func
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from atlazer.models.base import Base


class ChunkContextMetadata(BaseModel):
    user_id: str
    workspace_id: str
    content: str
    language_code: str = "en"
    chunks: Optional[list[Dict[str, Any]]] = None


class ContextChunkORM(Base):
    __tablename__ = "context_chunks"
    __table_args__ = (
        # 1. Vector Index (HNSW)
        Index(
            "idx_context_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),

        # 2. Filtering Indexes (B-Tree)
        Index("idx_context_chunks_workspace_id", "workspace_id"),
        Index("idx_context_chunks_user_id", "user_id"),
        Index("idx_context_chunks_context_id", "context_id"),
        Index("idx_context_chunks_workspace_context", "workspace_id", "context_id"),

        # 3. Metadata Index (GIN) - JSONB FILTERING
        Index(
            "idx_context_chunks_attributes",
            "attributes",
            postgresql_using="gin"
        ),

        # 4. Sorting Index
        Index("ix_context_chunks_context_chunk", "context_id", "chunk_index"),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    context_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
