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


class MatcherDepot:
    """
    Kumpulan operasi pencocokan (matching) paper berdasarkan kemiripan
    embedding minat user terhadap embedding chunk dokumen paper.

    Catatan implementasi (ASUMSI, sesuaikan bila skema berbeda):
      - `DocumentChunkORM.embedding` adalah kolom `Vector` (pgvector) sehingga
        punya method comparator `.cosine_distance(vector)`.
      - `DocumentChunkORM.paper_id` adalah FK ke `PaperORM.id`.
      - `DatabasePool.session()` adalah context manager sync yang
        menghasilkan objek `Session` SQLAlchemy.
    """

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def match_papers_by_interest(
        self,
        interest_embedder: List[float],
    ) -> List[PaperORM]:
        """
        Mencari paper yang embedding chunk-nya paling MIRIP dan paling TIDAK
        MIRIP dengan embedding minat user.

        Args:
            interest_embedder: vector embedding minat user (dari profile user).

        Returns:
            List[PaperORM]: berisi maksimal 2 item -> [paper_terdekat, paper_terjauh].
            - Jika hanya ada 1 paper unik di database, list hanya berisi 1 item
              (untuk menghindari paper duplikat di posisi terdekat & terjauh).
            - Jika tidak ada data sama sekali, mengembalikan list kosong.
        """
        try:
            with self._db_pool.session() as session:
                distance = DocumentChunkORM.embedding.cosine_distance(interest_embedder)

                closest_stmt = (
                    select(PaperORM, distance.label("distance"))
                    .join(DocumentChunkORM, DocumentChunkORM.paper_id == PaperORM.id)
                    .order_by(distance.asc())
                    .limit(1)
                )
                farthest_stmt = (
                    select(PaperORM, distance.label("distance"))
                    .join(DocumentChunkORM, DocumentChunkORM.paper_id == PaperORM.id)
                    .order_by(distance.desc())
                    .limit(1)
                )

                closest_row = session.execute(closest_stmt).first()
                farthest_row = session.execute(farthest_stmt).first()

                closest_paper = closest_row[0] if closest_row is not None else None
                farthest_paper = farthest_row[0] if farthest_row is not None else None

                result: List[PaperORM] = []
                if closest_paper is not None:
                    result.append(closest_paper)
                if farthest_paper is not None and (
                    closest_paper is None or farthest_paper.id != closest_paper.id
                ):
                    result.append(farthest_paper)

                logger.info(
                    "match_papers_by_interest -> found %d paper(s)", len(result)
                )
                return result

        except SQLAlchemyError:
            logger.exception(
                "Gagal melakukan pencocokan paper berdasarkan interest embedding"
            )
            raise
    