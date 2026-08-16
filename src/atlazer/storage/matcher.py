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
        candidate_pool_size: int = 100,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Mencari paper yang paling MIRIP ( closest ) dan paling TIDAK MIRIP

        ( farthest ) berdasarkan embedding minat user.

        Metode:
        - Closest: Menggunakan Weighted Scoring (Opsi B) -> `min_distance /
        LN(chunk_count + 1)`.
                Meminimalkan bobot agar paper dengan topik relevan mendalam
                mendapatkan peringkat teratas.
        - Farthest: Mencari paper yang bahkan chunk TERBAIK-nya memiliki jarak
        terjauh
                    (`min_distance DESC`), memastikan paper tersebut secara
                    keseluruhan
                    sama sekali tidak berkaitan dengan minat user.
        """
        results: Dict[str, List[Dict[str, Any]]] = {
            "closest": [],
            "farthest": [],
        }

        try:
            with self._db_pool.session() as session:
                distance_expr = DocumentChunkORM.embedding.cosine_distance(
                    intereset_embedding
                )

                # 1. Subquery paper yang sudah di-challenge oleh user (Eksklusi)
                challenged_subq = (
                    select(ChallengePaperORM.paper_id)
                    .join(
                        ChallengeORM, ChallengeORM.id == ChallengePaperORM.challenge_id
                    )
                    .where(ChallengeORM.user_id == user_id)
                )

                # -----------------------------------------------------------------
                # A. QUERY CLOSEST PAPER (Opsi B - Weighted Score)
                # -----------------------------------------------------------------
                closest_candidates = (
                    select(
                        DocumentChunkORM.paper_id.label("paper_id"),
                        distance_expr.label("distance"),
                    )
                    .where(DocumentChunkORM.paper_id.not_in(challenged_subq))
                    .order_by(distance_expr.asc())
                    .limit(candidate_pool_size)
                    .cte("closest_candidates")
                )

                closest_weighted = (
                    select(
                        closest_candidates.c.paper_id,
                        func.min(closest_candidates.c.distance).label(
                            "min_distance"
                        ),
                        func.count(closest_candidates.c.paper_id).label(
                            "chunk_count"
                        ),
                        (
                            func.min(closest_candidates.c.distance)
                            / func.ln(
                                func.count(closest_candidates.c.paper_id) + 1.0
                            )
                        ).label("weighted_score"),
                    )
                    .group_by(closest_candidates.c.paper_id)
                    .order_by(
                        (
                            func.min(closest_candidates.c.distance)
                            / func.ln(
                                func.count(closest_candidates.c.paper_id) + 1.0
                            )
                        ).asc()
                    )
                    .limit(1)
                ).cte("closest_weighted")

                closest_stmt = select(
                    PaperORM,
                    closest_weighted.c.min_distance,
                    closest_weighted.c.weighted_score,
                    closest_weighted.c.chunk_count,
                ).join(closest_weighted, PaperORM.id == closest_weighted.c.paper_id)

                closest_row = session.execute(closest_stmt).first()

                # -----------------------------------------------------------------
                # B. QUERY FARTHEST PAPER (Mencari paper yang paling bertolak belakang)
                # -----------------------------------------------------------------
                farthest_candidates = (
                    select(
                        DocumentChunkORM.paper_id.label("paper_id"),
                        distance_expr.label("distance"),
                    )
                    .where(DocumentChunkORM.paper_id.not_in(challenged_subq))
                    .order_by(distance_expr.desc())
                    .limit(candidate_pool_size)
                    .cte("farthest_candidates")
                )

                # Cari paper yang bahkan chunk terbaiknya paling jauh dari interest
                farthest_grouped = (
                    select(
                        farthest_candidates.c.paper_id,
                        func.min(farthest_candidates.c.distance).label(
                            "best_distance"
                        ),
                        func.max(farthest_candidates.c.distance).label(
                            "worst_distance"
                        ),
                    )
                    .group_by(farthest_candidates.c.paper_id)
                    .order_by(
                        func.min(farthest_candidates.c.distance).desc()
                    )  # min_distance paling besar
                    .limit(1)
                ).cte("farthest_grouped")

                farthest_stmt = select(
                    PaperORM,
                    farthest_grouped.c.best_distance,
                    farthest_grouped.c.worst_distance,
                ).join(
                    farthest_grouped, PaperORM.id == farthest_grouped.c.paper_id
                )

                farthest_row = session.execute(farthest_stmt).first()

                # -----------------------------------------------------------------
                # C. DRAFT DATA & EKSKLUSI DUPLIKASI
                # -----------------------------------------------------------------
                closest_paper_id = closest_row[0].id if closest_row else None
                farthest_paper_id = farthest_row[0].id if farthest_row else None

                # Hindari duplikasi jika farthest sama dengan closest
                if (
                    closest_paper_id
                    and farthest_paper_id
                    and closest_paper_id == farthest_paper_id
                ):
                    farthest_row = None
                    farthest_paper_id = None

                # -----------------------------------------------------------------
                # D. FETCH CHUNKS TERKAIT DALAM 1 QUERY
                # -----------------------------------------------------------------
                target_paper_ids = {
                    pid
                    for pid in (closest_paper_id, farthest_paper_id)
                    if pid is not None
                }
                chunks_by_paper_id: Dict[Any, List[Dict[str, Any]]] = {}

                if target_paper_ids:
                    chunks_stmt = (
                        select(
                            DocumentChunkORM,
                            distance_expr.label("distance"),
                        )
                        .where(DocumentChunkORM.paper_id.in_(target_paper_ids))
                        .order_by(
                            DocumentChunkORM.paper_id.asc(),
                            distance_expr.asc(),
                        )
                    )

                    for chunk_orm, dist in session.execute(chunks_stmt).all():
                        chunk_dict = {
                            column.name: getattr(chunk_orm, column.name)
                            for column in chunk_orm.__table__.columns
                        }
                        chunk_dict["distance"] = float(dist)
                        chunk_dict["similarity_score"] = float(1 - dist)
                        chunks_by_paper_id.setdefault(
                            chunk_orm.paper_id, []
                        ).append(chunk_dict)

                # -----------------------------------------------------------------
                # E. FORMAT HASIL (Mencegah DetachedInstanceError)
                # -----------------------------------------------------------------
                if closest_row:
                    paper_orm, min_dist, weighted_score, count = closest_row
                    paper_dict = {
                        column.name: getattr(paper_orm, column.name)
                        for column in paper_orm.__table__.columns
                    }
                    paper_dict["weighted_score"] = float(weighted_score)
                    paper_dict["matched_chunk_count"] = count

                    results["closest"].append(
                        {
                            "paper": paper_dict,
                            "distance": float(min_dist),
                            "relevance_score": float(1 - min_dist),
                            "chunks": chunks_by_paper_id.get(paper_orm.id, []),
                        }
                    )

                if farthest_row:
                    paper_orm, best_dist, worst_dist = farthest_row
                    paper_dict = {
                        column.name: getattr(paper_orm, column.name)
                        for column in paper_orm.__table__.columns
                    }

                    results["farthest"].append(
                        {
                            "paper": paper_dict,
                            "distance": float(best_dist),
                            "relevance_score": float(1 - best_dist),
                            "chunks": chunks_by_paper_id.get(paper_orm.id, []),
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