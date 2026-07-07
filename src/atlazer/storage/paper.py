"""Insert / upsert operations for the `papers` table (sync, SQLAlchemy)."""

from __future__ import annotations

import logging
from typing import List

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atlazer.storage.db import DatabasePool
from atlazer.models.paper import PaperCreate, PaperORM
from atlazer.models.document import DocumentChunkCreate, DocumentChunkORM

logger = logging.getLogger(__name__)


class PaperDuplicateError(Exception):
    """Raised when a paper can't be resolved to a single row via `(repository, identifier)` unique
    constraint."""


def _as_dict(item) -> dict:
    """Coerce a pydantic sub-model (author/affiliation) to a plain dict;
    pass through if it's already one."""
    return item.model_dump() if hasattr(item, "model_dump") else item


class PaperDepot:
    """Insert / upsert helpers for the `papers` table.

    Takes an already-started :class:`DatabasePool`.
    """

    # Name of the partial unique index/constraint on `doi`. Must match
    # exactly what's in the DB (name + WHERE predicate) or Postgres will
    # reject the ON CONFLICT clause at runtime — confirm this against
    # your migration before relying on it.
    _REPO_IDENTIFIER_COLUMNS = ["repository", "identifier"]
    _UPSERT_COLUMNS = (
        "repository", "identifier", "attributes",
        "title", "abstract", "year", "date_published",
        "authors", "affiliations",
        "venue", "venue_type", "publisher", "volume", "issue", "pages",
        "keywords", "fields_of_study", "language",
        "pdf_url", "open_access", "license",
        "references_count", "citations_count",
        "processing_tool", "processing_version", "processing_status",
        "error_message",
    )

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    def _values(self, paper: PaperCreate) -> dict:
        return {
            "doi": paper.doi or None,  # normalize '' -> NULL if it ever occurs
            "repository": paper.repository,
            "identifier": paper.identifier,
            "attributes": paper.attributes,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "date_published": paper.date_published,
            "authors": [_as_dict(a) for a in paper.authors],
            "affiliations": [_as_dict(a) for a in paper.affiliations],
            "venue": paper.venue,
            "venue_type": paper.venue_type,
            "publisher": paper.publisher,
            "volume": paper.volume,
            "issue": paper.issue,
            "pages": paper.pages,
            "keywords": paper.keywords,
            "fields_of_study": paper.fields_of_study,
            "language": paper.language,
            "pdf_url": paper.pdf_url,
            "open_access": paper.open_access,
            "license": paper.license,
            "references_count": paper.references_count,
            "citations_count": paper.citations_count,
            "processing_tool": paper.processing_tool,
            "processing_version": paper.processing_version,
            "processing_status": paper.processing_status,
            "error_message": paper.error_message,
        }

    def _upsert_stmt(self, values: dict, index_elements: list[str]):
        stmt = pg_insert(PaperORM).values(values)
        excluded = stmt.excluded
        set_ = {col: getattr(excluded, col) for col in self._UPSERT_COLUMNS}
        set_["updated_at"] = func.now()
        return stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=set_,
        ).returning(PaperORM.id)

    def upsert_paper(self, paper: PaperCreate, *, max_attempts: int = 3) -> str:
        """Insert a new paper, or update the matching row if one already
        exists under the same `(repository, identifier)`.

        Returns the `id` of the upserted row.
        """
        values = self._values(paper)
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            with self._pool.session() as session:
                try:
                    row = session.execute(
                        self._upsert_stmt(values, self._REPO_IDENTIFIER_COLUMNS)
                    ).fetchone()
                    session.commit()

                    if row is None:
                        # Arbiter resolved a conflict, but the conflicting row
                        # was concurrently deleted/rolled back before the
                        # UPDATE could apply (documented Postgres race for
                        # ON CONFLICT DO UPDATE ... RETURNING). Safe to retry
                        # the whole upsert from scratch.
                        logger.warning(
                            "repo+identifier upsert returned no row for repository=%s identifier=%s "
                            "(attempt %d/%d); retrying",
                            paper.repository, paper.identifier, attempt, max_attempts,
                        )
                        continue

                    logger.info(
                        "Upserted paper id=%s (matched on repository+identifier)", row.id
                    )
                    return str(row.id)

                except SQLAlchemyError as exc:
                    session.rollback()
                    last_exc = exc
                    logger.error(
                        "Database error during upsert for repository=%s identifier=%s: %s",
                        paper.repository, paper.identifier, exc,
                    )
                    # Jika terjadi error database selain konflik unik (misal: koneksi terputus),
                    # lemparkan errornya ke atas.
                    raise

        # Jika loop selesai tapi row masih None (kena race condition terus-menerus)
        raise PaperDuplicateError(
            f"Could not upsert paper "
            f"repository={paper.repository!r} identifier={paper.identifier!r} "
            f"after {max_attempts} attempts: no row returned by arbiter."
        ) from last_exc

    def bulk_insert_chunks(self, chunks: List[DocumentChunkCreate]) -> None:
        if not chunks:
            return

        # word_count / content_hash are GENERATED ALWAYS columns — never send them.
        values = [
            chunk.model_dump(exclude={"word_count", "content_hash"})
            for chunk in chunks
        ]

        stmt = pg_insert(DocumentChunkORM).values(values)

        # On (repository, identifier, section, chunk) collision, refresh the
        # mutable fields instead of aborting the whole batch — makes
        # re-processing a paper (crash/retry/duplicate task delivery) idempotent.
        update_cols = {
            "content": stmt.excluded.content,
            "chunk_type": stmt.excluded.chunk_type,
            "section_order": stmt.excluded.section_order,
            "embedding": stmt.excluded.embedding,
            "embedding_model": stmt.excluded.embedding_model,
            "embedding_adapter": stmt.excluded.embedding_adapter,
            "embedding_normalized": stmt.excluded.embedding_normalized,
            "token_count": stmt.excluded.token_count,
            "updated_at": func.now(),
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                DocumentChunkORM.repository,
                DocumentChunkORM.identifier,
                DocumentChunkORM.section,
                DocumentChunkORM.chunk,
            ],
            set_=update_cols,
        )

        with self._pool.session() as session:
            try:
                session.execute(stmt)
                session.commit()
                logger.info(
                    "Upserted %s chunks for repository=%s identifier=%s",
                    len(chunks), chunks[0].repository, chunks[0].identifier,
                )
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception("Failed to upsert chunks: %s", e)
                raise

    def get_last_paper(self, repository: str) -> PaperORM | None:
        """Return the most recently created paper row for `repository`,
        or `None` if that repository has no papers yet.

        "Newest" is determined by `created_at`; ties are broken by `id`
        so the result stays deterministic even if two rows share the
        same timestamp.
        """
        stmt = (
            select(PaperORM)
            .where(PaperORM.repository == repository)
            .order_by(PaperORM.created_at.desc(), PaperORM.id.desc())
            .limit(1)
        )

        with self._pool.session() as session:
            row = session.execute(stmt).scalar_one_or_none()
            if row is not None:
                # detach dari session supaya atribut tetap bisa diakses
                # setelah `with` block ini selesai (session close)
                session.expunge(row)
            return row