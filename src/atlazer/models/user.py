from datetime import timezone
from sqlalchemy import DateTime
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pydantic import field_validator, BaseModel, ConfigDict, Field
from pgvector.sqlalchemy import Vector
from .base import Base
from typing import List, Optional
from uuid import UUID


class ProfileORM(Base):
    __tablename__ = "profile"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    interest: Mapped[Optional[str]] = mapped_column(default="")
    interest_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    next_processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    language_code: Mapped[Optional[str]] = mapped_column(default="en")

    @field_validator('interest_embedding')
    def check_embedding_length(cls, v):
        if v is not None and len(v) != 1024:
            raise ValueError('Embedding must have exactly 1024 dimensions (vector(1024))')
        return v

    @field_validator('next_processed_at')
    def check_next_processed_at(cls, v):
        if v is not None and v < datetime.now(timezone.utc):
            raise ValueError('next_processed_at must be in the future')
        return v


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interest_embedding: Optional[List[float]] = Field(default=None)
    next_processed_at: Optional[datetime] = Field(default=None)

    @field_validator('interest_embedding')
    def check_embedding_length(cls, v):
        if v is not None and len(v) != 1024:
            raise ValueError('Embedding must have exactly 1024 dimensions (vector(1024))')
        return v
