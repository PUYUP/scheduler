from uuid import UUID
from sqlalchemy import Integer, String, Index, Float, func, UniqueConstraint
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from atlazer.models.base import Base


class ChunkBaseModel(BaseModel):
    user_id: str
    workspace_id: str
    content: str
    language_code: str = "en"
    chunks: Optional[list[Dict[str, Any]]] = None


class ChunkContextMetadata(ChunkBaseModel):
    context_id: str


class ChunkNoteMetadata(ChunkBaseModel):
    note_id: str


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

        # 5. Unique Constraint Index
        UniqueConstraint(
            "user_id", 
            "workspace_id", 
            "context_id", 
            "chunk_index", 
            name="uq_context_chunks_identity"
        ),
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


class ContextPaperORM(Base):
    __tablename__ = "context_papers"
    __table_args__ = (
        Index("idx_context_papers_context_paper", "context_id", "paper_id"),
        UniqueConstraint("context_id", "paper_id", name="uq_context_paper"),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    context_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    paper_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)


class ContextSimilarityORM(Base):
    __tablename__ = "context_similarities"
    __table_args__ = (
        Index("idx_context_sim_context_score", "context_id", "similarity_score"),
        Index("idx_context_sim_paper_score", "paper_id", "similarity_score"),
        Index("idx_context_similarities_attributes", "attributes", postgresql_using="gin"),
        UniqueConstraint(
            "context_chunk_id", 
            "document_chunk_id", 
            name="uq_context_doc_chunk_pair"
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    context_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    context_chunk_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    context_content: Mapped[str] = mapped_column(String, nullable=False)
    paper_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    document_chunk_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    document_content: Mapped[str] = mapped_column(String, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")


class ContextPaperSummaryORM(Base):
    __tablename__ = "context_papers_summaries"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    context_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    paper_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")


""" WORKSPACE NOTES """


class NoteChunkORM(Base):
    __tablename__ = "workspace_notes_chunks"
    __table_args__ = (
        # 1. Vector Index (HNSW)
        Index(
            "idx_workspace_notes_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),

        # 2. Filtering Indexes (B-Tree)
        Index("idx_workspace_notes_chunks_workspace_id", "workspace_id"),
        Index("idx_workspace_notes_chunks_user_id", "user_id"),
        Index("idx_workspace_notes_chunks_note_id", "note_id"),
        Index("idx_workspace_notes_chunks_workspace_note", "workspace_id", "note_id"),

        # 3. Metadata Index (GIN) - JSONB FILTERING
        Index(
            "idx_workspace_notes_chunks_attributes",
            "attributes",
            postgresql_using="gin"
        ),

        # 4. Sorting Index
        Index("ix_workspace_notes_chunks_note_chunk", "note_id", "chunk_index"),

        # 5. Unique Constraint Index
        UniqueConstraint(
            "user_id", 
            "workspace_id", 
            "note_id", 
            "chunk_index", 
            name="uq_workspace_notes_chunks_identity"
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    note_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")


class NotePaperORM(Base):
    __tablename__ = "workspace_notes_papers"
    __table_args__ = (
        Index("idx_workspace_note_papers_note_paper", "note_id", "paper_id"),
        UniqueConstraint("note_id", "paper_id", name="uq_workspace_note_paper"),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    note_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    paper_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)


class NoteSimilarityORM(Base):
    __tablename__ = "workspace_notes_similarities"
    __table_args__ = (
        Index("idx_workspace_notes_sim_note_score", "note_id", "similarity_score"),
        Index("idx_workspace_notes_sim_paper_score", "paper_id", "similarity_score"),
        Index("idx_workspace_notes_similarities_attributes", "attributes", postgresql_using="gin"),
        UniqueConstraint(
            "note_chunk_id", 
            "document_chunk_id", 
            name="uq_note_doc_chunk_pair"
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    workspace_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    note_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    note_chunk_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    note_content: Mapped[str] = mapped_column(String, nullable=False)
    paper_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    document_chunk_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    document_content: Mapped[str] = mapped_column(String, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
