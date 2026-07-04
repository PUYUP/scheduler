from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pydantic import field_validator, BaseModel, ConfigDict, Field
from pgvector.sqlalchemy import Vector
from .base import Base


class ProfileORM(Base):
    __tablename__ = "profile"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    interest_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=False)
    
    @field_validator('interest_embedding')
    def check_embedding_length(cls, v):
        if v is not None and len(v) != 1024:
            raise ValueError('Embedding must have exactly 1024 dimensions (vector(1024))')
        return v


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interest_embedding: Optional[List[float]] = Field(default=None)

    @field_validator('interest_embedding')
    def check_embedding_length(cls, v):
        if v is not None and len(v) != 1024:
            raise ValueError('Embedding must have exactly 1024 dimensions (vector(1024))')
        return v
