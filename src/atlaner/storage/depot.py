"""Insert / upsert operations for the `papers` table (sync, SQLAlchemy)."""

from __future__ import annotations

import logging

import orjson
from sqlalchemy import text, insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from atlaner.storage.db import DatabasePool
from atlaner.models.paper import PaperCreate
from atlaner.models.document import DocumentChunkCreate, DocumentChunkORM

logger = logging.getLogger(__name__)


class PaperDuplicateError(Exception):
    """Raised when a paper can't be resolved to a single row via either
    the `doi` UNIQUE constraint or the `(repository, identifier)` UNIQUE
    constraint."""


class PaperDepot:
    """Insert / upsert helpers for the `papers` table.

    Takes an already-started :class:`DatabasePool`.
    """

    _INSERT_ON_DOI_CONFLICT_SQL = """
        INSERT INTO papers (
            doi, repository, identifier, metadata,
            title, abstract, year, date_published,
            authors, affiliations,
            venue, venue_type, publisher, volume, issue, pages,
            keywords, fields_of_study, language,
            pdf_url, open_access, license,
            references_count, citations_count,
            processing_tool, processing_version, processing_status, error_message
        ) VALUES (
            :doi, :repository, :identifier, :metadata,
            :title, :abstract, :year, :date_published,
            :authors, :affiliations,
            :venue, :venue_type, :publisher, :volume, :issue, :pages,
            :keywords, :fields_of_study, :language,
            :pdf_url, :open_access, :license,
            :references_count, :citations_count,
            :processing_tool, :processing_version, :processing_status, :error_message
        )
        ON CONFLICT (doi) WHERE doi IS NOT NULL DO UPDATE SET
            repository           = EXCLUDED.repository,
            identifier           = EXCLUDED.identifier,
            metadata             = EXCLUDED.metadata,
            title                = EXCLUDED.title,
            abstract             = EXCLUDED.abstract,
            year                 = EXCLUDED.year,
            date_published       = EXCLUDED.date_published,
            authors              = EXCLUDED.authors,
            affiliations         = EXCLUDED.affiliations,
            venue                = EXCLUDED.venue,
            venue_type           = EXCLUDED.venue_type,
            publisher            = EXCLUDED.publisher,
            volume               = EXCLUDED.volume,
            issue                = EXCLUDED.issue,
            pages                = EXCLUDED.pages,
            keywords             = EXCLUDED.keywords,
            fields_of_study      = EXCLUDED.fields_of_study,
            language             = EXCLUDED.language,
            pdf_url              = EXCLUDED.pdf_url,
            open_access          = EXCLUDED.open_access,
            license              = EXCLUDED.license,
            references_count     = EXCLUDED.references_count,
            citations_count      = EXCLUDED.citations_count,
            processing_tool      = EXCLUDED.processing_tool,
            processing_version   = EXCLUDED.processing_version,
            processing_status    = EXCLUDED.processing_status,
            error_message        = EXCLUDED.error_message,
            updated_at           = NOW()
        RETURNING id;
    """

    # Fallback: matched on (repository, identifier) instead of doi.
    _UPDATE_ON_REPO_IDENTIFIER_SQL = """
        UPDATE papers SET
            doi                  = :doi,
            metadata             = :metadata,
            title                = :title,
            abstract             = :abstract,
            year                 = :year,
            date_published       = :date_published,
            authors              = :authors,
            affiliations         = :affiliations,
            venue                = :venue,
            venue_type           = :venue_type,
            publisher            = :publisher,
            volume               = :volume,
            issue                = :issue,
            pages                = :pages,
            keywords             = :keywords,
            fields_of_study      = :fields_of_study,
            language             = :language,
            pdf_url              = :pdf_url,
            open_access          = :open_access,
            license              = :license,
            references_count     = :references_count,
            citations_count      = :citations_count,
            processing_tool      = :processing_tool,
            processing_version   = :processing_version,
            processing_status    = :processing_status,
            error_message        = :error_message,
            updated_at           = NOW()
        WHERE repository = :repository AND identifier = :identifier
        RETURNING id;
    """

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    # Insert/update paper
    def upsert_paper(self, paper: PaperCreate) -> str:
        """Insert a new paper, or update the matching row if one already
        exists under the same `doi` OR the same `(repository, identifier)`.

        Returns the `id` of the upserted row.

        Raises:
            PaperDuplicateError: if the row can't be resolved to a single
                match.
        """
        params = {
            "doi": paper.doi,
            "repository": paper.repository,
            "identifier": paper.identifier,
            "metadata": orjson.dumps(paper.metadata).decode(),
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "date_published": paper.date_published,
            "authors": orjson.dumps(list(paper.authors)).decode(),
            "affiliations": orjson.dumps(list(paper.affiliations)).decode(),
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

        with self._pool.session() as session:
            try:
                row = session.execute(
                    text(self._INSERT_ON_DOI_CONFLICT_SQL), params
                ).fetchone()
                session.commit()
                logger.info("Upserted paper id=%s (matched on doi)", row.id)
                return str(row.id)

            except IntegrityError as exc:
                # `doi` arbiter let the INSERT through, but the other
                # UNIQUE constraint (repository, identifier) fired.
                session.rollback()
                logger.info(
                    "doi=%s not found; falling back to (repository=%s, identifier=%s): %s",
                    paper.doi, paper.repository, paper.identifier, exc,
                )
                try:
                    row = session.execute(
                        text(self._UPDATE_ON_REPO_IDENTIFIER_SQL), params
                    ).fetchone()
                    session.commit()
                except IntegrityError as exc2:
                    session.rollback()
                    logger.warning(
                        "Irreconcilable conflict for doi=%s repository=%s/%s: %s",
                        paper.doi, paper.repository, paper.identifier, exc2,
                    )
                    raise PaperDuplicateError(
                        f"Paper with doi={paper.doi!r} conflicts with a "
                        f"different row than the one matched by "
                        f"repository={paper.repository!r}/identifier={paper.identifier!r}."
                    ) from exc2

                if row is None:
                    raise PaperDuplicateError(
                        f"Could not upsert paper doi={paper.doi!r} "
                        f"repository={paper.repository!r} identifier={paper.identifier!r}: "
                        f"no existing row matched either unique key."
                    ) from exc

                logger.info(
                    "Upserted paper id=%s (matched on repository+identifier)", row.id
                )
                return str(row.id)

    # Bulk insert chunks
    def bulk_insert_chunks(self, chunks: List[DocumentChunkCreate]) -> None:
        if not chunks:
            return

        # word_count / content_hash are GENERATED ALWAYS columns — never send them.
        values = [
            chunk.model_dump(exclude={"word_count", "content_hash"})
            for chunk in chunks
        ]

        with self._pool.session() as session:
            try:
                session.execute(insert(DocumentChunkORM), values)
                session.commit()
                logger.info(
                    "Inserted %s chunks for paper %s", len(chunks), chunks[0].paper_id
                )
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception("Failed to insert chunks: %s", e)
                raise
