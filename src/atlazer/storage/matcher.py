"""Insert / upsert operations for the `papers` table (sync, SQLAlchemy)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from atlazer.storage.db import DatabasePool
from atlazer.models.paper import PaperORM
from atlazer.models.document import DocumentChunkORM
from atlazer.models.challenge import ChallengePaperORM, ChallengeORM

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
        user_id: str,
        intereset_embedding: List[float],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Mencari paper yang embedding chunk-nya paling MIRIP dan paling TIDAK
        MIRIP dengan embedding minat user, beserta relevance score-nya.

        Args:
            user_id: UUID dari user yang sedang dicarikan paper (dalam bentuk string).
            intereset_embedding: vector embedding minat user (dari profile user).

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
                distance = DocumentChunkORM.embedding.cosine_distance(intereset_embedding)

                # Subquery untuk paper yang sudah dichallenge user
                challenged_subq = (
                    select(ChallengePaperORM.paper_id)
                    .join(ChallengeORM, ChallengeORM.id == ChallengePaperORM.challenge_id)
                    .where(ChallengeORM.user_id == user_id)
                )

                # Query untuk mencari paper yang paling mirip, mengecualikan yang sudah ada
                closest_stmt = (
                    select(PaperORM, distance.label("distance"))
                    .join(DocumentChunkORM, DocumentChunkORM.paper_id == PaperORM.id)
                    .where(PaperORM.id.not_in(challenged_subq))
                    .order_by(distance.asc())
                    .limit(1)
                )

                # Query untuk mencari paper yang paling tidak mirip, mengecualikan yang sudah ada
                farthest_stmt = (
                    select(PaperORM, distance.label("distance"))
                    .join(DocumentChunkORM, DocumentChunkORM.paper_id == PaperORM.id)
                    .where(PaperORM.id.not_in(challenged_subq))
                    .order_by(distance.desc())
                    .limit(1)
                )

                closest_row = session.execute(closest_stmt).first()
                farthest_row = session.execute(farthest_stmt).first()

                closest_paper = closest_row[0] if closest_row is not None else None
                closest_distance = closest_row[1] if closest_row is not None else None

                farthest_paper = farthest_row[0] if farthest_row is not None else None
                farthest_distance = farthest_row[1] if farthest_row is not None else None

                # Ambil semua chunk untuk paper_id yang relevan dalam SATU query,
                # lalu kelompokkan per paper_id di sisi Python.
                paper_ids = {p.id for p in (closest_paper, farthest_paper) if p is not None}

                chunks_by_paper_id: Dict[Any, List[DocumentChunkORM]] = {}
                if paper_ids:
                    chunks_stmt = (
                        select(DocumentChunkORM)
                        .where(DocumentChunkORM.paper_id.in_(paper_ids))
                        .order_by(
                            DocumentChunkORM.paper_id.asc(),
                            DocumentChunkORM.id.asc(),  # ganti ke chunk_index jika ada
                        )
                    )
                    for chunk in session.execute(chunks_stmt).scalars().all():
                        chunks_by_paper_id.setdefault(chunk.paper_id, []).append(chunk)

                results: Dict[str, List[Dict[str, Any]]] = {
                    "closest": [],
                    "farthest": [],
                }

                if closest_paper is not None and closest_distance is not None:
                    results["closest"].append(
                        {
                            "paper": closest_paper,
                            "distance": closest_distance,
                            "relevance_score": 1 - closest_distance,
                            "chunks": chunks_by_paper_id.get(closest_paper.id, []),
                        }
                    )

                if farthest_paper is not None and farthest_distance is not None and (
                    closest_paper is None or farthest_paper.id != closest_paper.id
                ):
                    results["farthest"].append(
                        {
                            "paper": farthest_paper,
                            "distance": farthest_distance,
                            "relevance_score": 1 - farthest_distance,
                            "chunks": chunks_by_paper_id.get(farthest_paper.id, []),
                        }
                    )

                logger.info(
                    "match_papers_by_interest -> closest=%d farthest=%d",
                    len(results["closest"]),
                    len(results["farthest"]),
                )
                return results

        except SQLAlchemyError:
            logger.exception(
                "Gagal melakukan pencocokan paper berdasarkan interest embedding"
            )
            raise