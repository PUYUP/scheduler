"""Pydantic models for the `papers` table.

Mirrors the DDL in `storage/db.py`. Split into three models:

* PaperCreate  — payload for INSERT. Excludes DB-generated columns
  (`id`, `created_at`, `updated_at`). Columns with a SQL DEFAULT get a
  matching Python default so callers can omit them.
* PaperUpdate  — payload for PATCH-style partial updates. Every field
  is optional; only fields explicitly set should be sent to the DB
  (use `.model_dump(exclude_unset=True)`).
* PaperRead    — full row as returned from the DB, including the
  read-only / auto-generated columns.

NOTE: `repository_identifier` / `repository_metadata` were renamed to
`identifier` / `metadata` to match the current DB schema.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    Text, UniqueConstraint, func
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from .base import Base


# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------


class PaperORM(Base):
    """
    Maps 1:1 onto the `papers` table defined in the DDL.
    Must share the same `Base` (and therefore the same MetaData) as
    DocumentChunkORM below, otherwise ForeignKey("papers.id") cannot
    be resolved and SQLAlchemy raises NoReferencedTableError.
    """
    __tablename__ = "papers"
 
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
 
    identifier: Mapped[str] = mapped_column(Text, nullable=False)
    repository: Mapped[str] = mapped_column(Text, nullable=False)
    # repository_metadata, title, abstract, authors, etc. omitted here for
    # brevity — add the remaining columns from the DDL as needed. Only the
    # primary key matters for the FK to resolve; SQLAlchemy doesn't require
    # every column to be mapped for cross-table FK references to work.
 
    __table_args__ = (
        UniqueConstraint("repository", "identifier", name="idx_papers_repo_identifier"),
    )


# ---------------------------------------------------------------------------
# Nested JSONB shapes
# ---------------------------------------------------------------------------


class Affiliation(BaseModel):
    """One entry of the `affiliations` JSONB array."""

    name: str | None = None
    department: str | None = None
    country: str | None = None


# ---------------------------------------------------------------------------
# CHECK-constrained literals (kept in sync with the DDL)
# ---------------------------------------------------------------------------

VenueType = Literal["journal", "conference", "preprint", "book", "thesis", "report"]
ProcessingStatus = Literal["pending", "indexing", "done", "failed"]


# ---------------------------------------------------------------------------
# Create — insert payload
# ---------------------------------------------------------------------------


class PaperCreate(BaseModel):
  """Payload for creating a new paper row.

  Excludes: `id`, `created_at`, `updated_at` (DB auto-generated).
  """

  model_config = ConfigDict(extra="forbid")

  # `id` (BIGSERIAL) is the sole identifier; it's DB auto-generated,
  # so it's intentionally absent here.

  # Required (NOT NULL, no default) ------------------------------------
  doi: str | None = None
  repository: str
  identifier: str
  title: str

  # Optional with DB default — mirrored here so callers can omit them --
  metadata: dict = Field(default_factory=dict)
  authors: list[str] = Field(default_factory=list)
  affiliations: list[Affiliation] = Field(default_factory=list)
  language: str = "en"
  processing_status: ProcessingStatus = "pending"

  # Optional, nullable in DB --------------------------------------------
  abstract: str | None = None
  year: int | None = None
  date_published: date | None = None

  venue: str | None = None
  venue_type: VenueType | None = None
  publisher: str | None = None
  volume: str | None = None
  issue: str | None = None
  pages: str | None = None

  keywords: list[str] | None = None
  fields_of_study: list[str] | None = None

  pdf_url: str | None = None
  open_access: bool | None = None
  license: str | None = None

  references_count: int | None = None
  citations_count: int | None = None

  processing_tool: str | None = None
  processing_version: str | None = None
  error_message: str | None = None

  @field_validator("doi", mode="before")
  @classmethod
  def empty_doi_to_none(cls, v):
    return v or None


# ---------------------------------------------------------------------------
# Update — partial patch payload
# ---------------------------------------------------------------------------


class PaperUpdate(BaseModel):
  """Payload for partial updates (PATCH semantics).

  Every field is optional. Use `.model_dump(exclude_unset=True)` when
  building the SQL SET clause so untouched fields aren't overwritten.
  Still excludes `id`, `created_at`, `updated_at`.
  """

  model_config = ConfigDict(extra="forbid")

  # `id` is server-generated and immutable; never part of an update
  # payload.
  doi: str | None = None
  repository: str | None = None
  identifier: str | None = None
  metadata: dict | None = None

  title: str | None = None
  abstract: str | None = None
  year: int | None = None
  date_published: date | None = None

  authors: list[str] | None = None
  affiliations: list[Affiliation] | None = None

  venue: str | None = None
  venue_type: VenueType | None = None
  publisher: str | None = None
  volume: str | None = None
  issue: str | None = None
  pages: str | None = None

  keywords: list[str] | None = None
  fields_of_study: list[str] | None = None
  language: str | None = None

  pdf_url: str | None = None
  open_access: bool | None = None
  license: str | None = None

  references_count: int | None = None
  citations_count: int | None = None

  processing_tool: str | None = None
  processing_version: str | None = None
  processing_status: ProcessingStatus | None = None
  error_message: str | None = None


# ---------------------------------------------------------------------------
# Read — full row as returned from the DB
# ---------------------------------------------------------------------------


class PaperRead(BaseModel):
  """Full row shape, including DB-generated / read-only columns."""

  model_config = ConfigDict(from_attributes=True)

  # Read-only / auto-generated -------------------------------------------
  id: int
  created_at: datetime
  updated_at: datetime

  # Same as PaperCreate below this point ----------------------------------
  doi: str
  repository: str
  identifier: str
  metadata: dict
  title: str
  abstract: str | None = None
  year: int | None = None
  date_published: date | None = None

  authors: list[str] = Field(default_factory=list)
  affiliations: list[Affiliation] = Field(default_factory=list)

  venue: str | None = None
  venue_type: VenueType | None = None
  publisher: str | None = None
  volume: str | None = None
  issue: str | None = None
  pages: str | None = None

  keywords: list[str] | None = None
  fields_of_study: list[str] | None = None
  language: str = "en"

  pdf_url: str | None = None
  open_access: bool | None = None
  license: str | None = None

  references_count: int | None = None
  citations_count: int | None = None

  processing_tool: str | None = None
  processing_version: str | None = None
  processing_status: ProcessingStatus = "pending"
  error_message: str | None = None
