"""Insert / upsert operations for the `papers` table (sync, SQLAlchemy)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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
      - `cosine_distance` mengembalikan jarak (0 = identik). `relevance_score`
        dihitung sebagai `1 - distance` (cosine similarity) dengan asumsi
        embedding sudah dinormalisasi. Sesuaikan formula ini bila tidak.
    """

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def match_papers_by_interest(
        self,
        interest_embedder: List[float],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Mencari paper yang embedding chunk-nya paling MIRIP dan paling TIDAK
        MIRIP dengan embedding minat user, beserta relevance score-nya.

        Args:
            interest_embedder: vector embedding minat user (dari profile user).

        Returns:
            Dict[str, List[Dict[str, Any]]]: dict dengan 2 key -> "closest" dan
            "farthest". Tiap key berisi list maksimal 1 item dengan struktur:
                {
                    "paper": PaperORM,
                    "distance": float,          # cosine distance mentah
                    "relevance_score": float,   # 1 - distance
                }
            - "closest": paper paling mirip (list kosong jika tidak ada data).
            - "farthest": paper paling tidak mirip. Jika hanya ada 1 paper unik
              di database (closest == farthest), "farthest" dikembalikan
              sebagai list kosong untuk menghindari duplikasi paper yang sama.
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
                closest_distance = closest_row[1] if closest_row is not None else None

                farthest_paper = farthest_row[0] if farthest_row is not None else None
                farthest_distance = farthest_row[1] if farthest_row is not None else None

                result: Dict[str, List[Dict[str, Any]]] = {
                    "closest": [],
                    "farthest": [],
                }

                if closest_paper is not None:
                    result["closest"].append(
                        {
                            "paper": closest_paper,
                            "distance": closest_distance,
                            "relevance_score": 1 - closest_distance,
                        }
                    )

                if farthest_paper is not None and (
                    closest_paper is None or farthest_paper.id != closest_paper.id
                ):
                    result["farthest"].append(
                        {
                            "paper": farthest_paper,
                            "distance": farthest_distance,
                            "relevance_score": 1 - farthest_distance,
                        }
                    )

                logger.info(
                    "match_papers_by_interest -> closest=%d farthest=%d",
                    len(result["closest"]),
                    len(result["farthest"]),
                )
                return result

        except SQLAlchemyError:
            logger.exception(
                "Gagal melakukan pencocokan paper berdasarkan interest embedding"
            )
            raise