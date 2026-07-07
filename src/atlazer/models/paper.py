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
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import (
    TIMESTAMP, Integer, Boolean, ARRAY, Date, SmallInteger, Text, UniqueConstraint, func
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .base import Base


# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------


class PaperORM(Base):
    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(
      PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )

    doi: Mapped[str | None] = mapped_column(Text)
    repository: Mapped[str] = mapped_column(Text, nullable=False)
    identifier: Mapped[str] = mapped_column(Text, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(SmallInteger)
    date_published: Mapped[Date | None] = mapped_column(Date)

    authors: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    affiliations: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")

    venue: Mapped[str | None] = mapped_column(Text)
    venue_type: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    volume: Mapped[str | None] = mapped_column(Text)
    issue: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[str | None] = mapped_column(Text)

    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    fields_of_study: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    language: Mapped[str] = mapped_column(Text, nullable=False, server_default="en")

    pdf_url: Mapped[str | None] = mapped_column(Text)
    open_access: Mapped[bool | None] = mapped_column(Boolean)
    license: Mapped[str | None] = mapped_column(Text)

    references_count: Mapped[int | None] = mapped_column(Integer)
    citations_count: Mapped[int | None] = mapped_column(Integer)

    processing_tool: Mapped[str | None] = mapped_column(Text)
    processing_version: Mapped[str | None] = mapped_column(Text)
    processing_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)

    last_scraped_category: Mapped[str | None] = mapped_column(Text)
    last_scraped_page: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[TIMESTAMP] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[TIMESTAMP] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("repository", "identifier", name="idx_papers_repo_identifier"),
        # If a unique index on `doi` exists in the live DB (required for the
        # doi-arbiter upsert below), declare it here too, e.g.:
        # Index("idx_papers_doi_unique", "doi", unique=True,
        #       postgresql_where=text("doi IS NOT NULL")),
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
  attributes: dict = Field(default_factory=dict)
  title: str

  # Optional with DB default — mirrored here so callers can omit them --
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

  last_scraped_category: str | None = None
  last_scraped_page: str | None = None

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
  attributes: dict | None = None
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

  last_scraped_category: str | None = None
  last_scraped_page: str | None = None


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
  attributes: dict
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

  last_scraped_category: str | None = None
  last_scraped_page: str | None = None
