from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from atlazer.storage.db import DatabasePool
from atlazer.models.paper import ScrapeProgressORM

logger = logging.getLogger(__name__)


class ScrapeProgressDepot:
    """Read / upsert helpers for the `scrape_progress` table.

    Takes an already-started :class:`DatabasePool`.
    """

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    def get_progress(self, repository: str, topic: str) -> ScrapeProgressORM | None:
        """Return the progress row for this exact (repository, topic) pair,
        or `None` if no progress has been recorded yet.

        `(repository, topic)` is unique-constrained at the DB level, so
        this can never match more than one row.
        """
        stmt = select(ScrapeProgressORM).where(
            ScrapeProgressORM.repository == repository,
            ScrapeProgressORM.topic == topic,
        )

        with self._pool.session() as session:
            try:
                row = session.execute(stmt).scalar_one_or_none()
            except SQLAlchemyError:
                logger.exception(
                    "Failed to fetch scrape progress for repository=%s topic=%s",
                    repository, topic,
                )
                raise

            if row is not None:
                # detach dari session supaya atribut tetap bisa diakses
                # setelah `with` block ini selesai (session close)
                session.expunge(row)
            return row

    def get_start_offset(self, repository: str, topic: str, default: int = 0) -> int:
        """Convenience wrapper around `get_progress` for callers that only
        need the numeric offset (e.g. `scrape_topic_increment`), without
        having to handle the `None` / detached-ORM-object case themselves.
        """
        row = self.get_progress(repository, topic)
        return row.start_offset if row is not None else default

    def set_progress(
        self,
        repository: str,
        topic: str,
        start_offset: int,
    ) -> ScrapeProgressORM:
        """Upsert progress untuk kombinasi (repository, topic).

        Jika baris untuk (repository, topic) sudah ada, `start_offset`-nya
        akan di-update. Jika belum ada, baris baru akan dibuat.
        """
        stmt = select(ScrapeProgressORM).where(
            ScrapeProgressORM.repository == repository,
            ScrapeProgressORM.topic == topic,
        )

        with self._pool.session() as session:
            try:
                row = session.execute(stmt).scalar_one_or_none()

                if row is not None:
                    # sudah ada -> update saja
                    row.start_offset = start_offset
                else:
                    # belum ada -> buat baris baru
                    row = ScrapeProgressORM(
                        repository=repository,
                        topic=topic,
                        start_offset=start_offset,
                    )
                    session.add(row)

                session.commit()
                session.refresh(row)
            except SQLAlchemyError:
                logger.exception(
                    "Failed to upsert scrape progress for repository=%s topic=%s",
                    repository, topic,
                )
                session.rollback()
                raise

            # detach dari session supaya atribut tetap bisa diakses
            # setelah `with` block ini selesai (session close)
            session.expunge(row)
            return row
