from pydantic import BaseModel, conint, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Text, Integer, Boolean, TIMESTAMP, CheckConstraint,
    UniqueConstraint, ForeignKey, func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector

from .base import Base

# Definisi Literal untuk Enum chunk_type
ChunkType = Literal['abstract', 'body', 'conclusion', 'caption', 'equation', 'other']

class DocumentChunkBase(BaseModel):
    """
    Model dasar yang berisi field utama yang diisi oleh aplikasi (user/system).
    """
    paper_id: str
    repository: str
    identifier: str
    section: str
    section_order: str
    chunk: str
    chunk_type: ChunkType = 'body'
    content: str
    
    # Embedding fields (biasanya opsional saat awal dibuat, diisi oleh worker)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_adapter: Optional[str] = None
    embedding_normalized: bool = True
    token_count: Optional[conint(gt=0)] = None # constraint: > 0 jika tidak None

    @field_validator('embedding')
    def check_embedding_length(cls, v):
        if v is not None and len(v) != 1024:
            raise ValueError('Embedding must have exactly 1024 dimensions (vector(1024))')
        return v

    @field_validator('content')
    def check_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class DocumentChunkCreate(DocumentChunkBase):
    """
    Model untuk validasi data SEBELUM masuk ke database.
    (Biasanya tidak memerlukan id, created_at, dll karena digenerate oleh DB)
    """
    pass


# ---------------------------------------------------------------------------
# SQLAlchemy ORM model (the actual mapped table — what insert() needs)
# ---------------------------------------------------------------------------


class DocumentChunkORM(Base):
    """
    Maps 1:1 onto the `document_chunks` table defined in the DDL.
    This is the class you pass to `insert(...)` / `session.add(...)`,
    NOT the Pydantic `DocumentChunk` above.
    """
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )

    paper_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )

    repository: Mapped[str] = mapped_column(Text, nullable=False)
    identifier: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)
    section_order: Mapped[int] = mapped_column(Text, nullable=False)
    chunk: Mapped[int] = mapped_column(Text, nullable=False)

    chunk_type: Mapped[str] = mapped_column(Text, nullable=False, server_default="body")
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # GENERATED ALWAYS AS (...) STORED columns — read-only from the app's
    # perspective. Do NOT set these on insert; let Postgres compute them.
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, insert_default=None)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False, insert_default=None)

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_adapter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_normalized: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("paper_id", "section", "chunk", name="document_chunks_paper_id_section_chunk_key"),
        CheckConstraint(
            "chunk_type IN ('abstract','body','conclusion','caption','equation','other')",
            name="chunks_type_check",
        ),
        CheckConstraint("token_count IS NULL OR token_count > 0", name="chunks_token_count_positive"),
        CheckConstraint("word_count > 0", name="chunks_word_count_positive"),
    )
