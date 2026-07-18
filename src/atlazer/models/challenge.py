import secrets
from uuid import UUID
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, conint, field_validator

from sqlalchemy import (
    event, 
    func, 
    String, 
    Date,
    Numeric, 
    Boolean,
    Integer,
    Text, 
    ForeignKey,
    UniqueConstraint, 
    CheckConstraint,
    TIMESTAMP
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from atlazer.models.base import Base


class ChallengeORM(Base):
    """
    Model ini menyimpan hasil matching dari paper menjadi sebuah challenge.
    """
    __tablename__ = "challenges"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    code: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    target_date: Mapped[Date] = mapped_column(Date, nullable=False)

    # Relasi ke ChallengePaperORM (Otomatis mengambil data paper yang terhubung)
    # cascade="all, delete-orphan" memastikan konsistensi object di memory saat parent dihapus
    challenge_papers: Mapped[List["ChallengePaperORM"]] = relationship(
        back_populates="challenge", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )


class ChallengePaperORM(Base):
    """
    Junction table yang memetakan challenge ke paper-paper yang direkomendasikan.
    """
    __tablename__ = "challenge_papers"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    challenge_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        ForeignKey("challenges.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )
    # Asumsi tabel paper Anda bernama "papers", sesuaikan jika berbeda.
    paper_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    # Optional[Decimal] karena mungkin ada kasus nilai ini kosong sementara, 
    # hapus Optional jika wajib diisi dari awal.
    relevance_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2))
    relevance_label: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(Text, server_default="unread")

    # Relasi balik ke ChallengeORM
    challenge: Mapped["ChallengeORM"] = relationship(back_populates="challenge_papers")
    
    # Jika Anda memiliki PaperORM, Anda bisa uncomment dan tambahkan relasi ke sini:
    # paper: Mapped["PaperORM"] = relationship()

    __table_args__ = (
        UniqueConstraint("challenge_id", "paper_id", name="unique_challenge_paper"),
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="check_relevance_score"),
        CheckConstraint("relevance_label IN ('closest', 'farthest')", name="check_relevance_label"),
    )


class PaperSummaryORM(Base):
    __tablename__ = "paper_summaries"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    paper_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    # optional relation to the challenge entity
    challenge_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        index=True,
        nullable=True
    )
    challenge_paper_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("challenge_papers.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
        unique=True
    )
    
    tool: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    results: Mapped[Optional[JSONB]] = mapped_column(JSONB, nullable=True)
    job_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    created_at: Mapped[TIMESTAMP] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    finished_at: Mapped[Optional[TIMESTAMP]] = mapped_column(
        TIMESTAMP(timezone=True)
        # tidak ada default -> tetap NULL sampai di-set eksplisit
    )
    updated_at: Mapped[TIMESTAMP] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    __table_args__ = (
        UniqueConstraint("challenge_id", "paper_id", name="unique_challenge_paper"),
    )


@event.listens_for(ChallengeORM, "before_insert")
def generate_code_before_insert(mapper, connection, target: ChallengeORM):
    if target.code:
        return  # jangan overwrite kalau sudah di-set manual

    user_id_str = str(target.user_id)
    uuid_chars = user_id_str.replace("-", "").upper()
    first_three = uuid_chars[:3]
    random_index = secrets.randbelow(len(uuid_chars) - 1)
    prefix = uuid_chars[random_index:random_index + 2]
    random_part = secrets.token_hex(4).upper()
    target.code = f"{first_three}-{prefix}-{random_part}"


class ChunkAnswerMetadata(BaseModel):
    user_id: str
    challenge_id: str
    answer_id: str
    content: str
    language_code: Optional[str] = None
    chunks: Optional[list[Dict[str, Any]]] = None


class AnswerChunkORM(Base):
    __tablename__ = "answer_chunks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    user_id:        Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    challenge_id:   Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    answer_id:      Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    content:        Mapped[str] = mapped_column(Text, nullable=False)

    embedding:      Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    embedding_adapter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    embedding_normalized: Mapped[bool] = mapped_column(Boolean, default=True)
    token_count: Mapped[Optional[conint(gt=0)]] = mapped_column(Integer, nullable=True)
    word_count: Mapped[Optional[conint(gt=0)]] = mapped_column(Integer, nullable=True)

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
