import uuid
import structlog

from uuid import UUID
from atlazer.models.paper import PaperORM

from typing import List, Any, Dict, TypedDict, Optional
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import NoteChunkORM, NotePaperORM, NoteSimilarityORM
from atlazer.models.document import DocumentChunkORM

from sqlalchemy import tuple_, insert, select, func
from sqlalchemy.exc import SQLAlchemyError

log = structlog.get_logger()


# --- 1. Type Hinting Setup ---
class SimilarChunkDict(TypedDict):
    document_content: str
    document_id: Any
    chunk_content: str
    chunk_id: Any
    paper_id: Any
    similarity_score: float


class MatchResultDict(TypedDict):
    id: Any
    chunk_content: str
    papers: List[Dict[str, Any]]  # Berisi list of dictionary dari paper
    similar_chunks: List[SimilarChunkDict]


class WorkspaceNoteDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def bulk_insert_chunks(self, values: List[NoteChunkORM]) -> None:
        if not values:
            log.info("workspace.bulk_insert_chunks.empty_list")
            return

        note_keys = list({
            (chunk.user_id, chunk.note_id, chunk.workspace_id) 
            for chunk in values
        })

        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "note_id": chunk.note_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "attributes": chunk.attributes,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(NoteChunkORM).filter(
                    tuple_(
                        NoteChunkORM.user_id, 
                        NoteChunkORM.note_id,
                        NoteChunkORM.workspace_id,
                    ).in_(note_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(NoteChunkORM), rows)
                session.commit()

                log.info(
                    "workspace.bulk_insert_note_chunks.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_chunks.error_upsert",
                    error=str(e),
                )
                raise e

    def bulk_insert_papers(self, values: List[NotePaperORM]) -> None:
        if not values:
            log.info("workspace.bulk_insert_note_papers.empty_list")
            return

        note_keys = list({(chunk.user_id, chunk.note_id) for chunk in values})
        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "note_id": chunk.note_id,
                "paper_id": chunk.paper_id,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(NotePaperORM).filter(
                    tuple_(
                        NotePaperORM.user_id, 
                        NotePaperORM.note_id
                    ).in_(note_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(NotePaperORM), rows)
                session.commit()

                log.info(
                    "workspace.bulk_insert_notes_papers.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_notes_papers.error_upsert",
                    error=str(e),
                )
                raise e

    def bulk_insert_similarities(
        self,
        values: List[NoteSimilarityORM]
    ) -> None:
        if not values:
            log.info("workspace.bulk_insert_notes_similarities.empty_list")
            return

        note_keys = list({(chunk.user_id, chunk.note_id, chunk.workspace_id, chunk.note_chunk_id) for chunk in values})
        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "note_id": chunk.note_id,
                "note_chunk_id": chunk.note_chunk_id,
                "note_content": chunk.note_content,
                "paper_id": chunk.paper_id,
                "document_chunk_id": chunk.document_chunk_id,
                "document_content": chunk.document_content,
                "attributes": chunk.attributes,
                "similarity_score": chunk.similarity_score,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(NoteSimilarityORM).filter(
                    tuple_(
                        NoteSimilarityORM.user_id, 
                        NoteSimilarityORM.note_id,
                        NoteSimilarityORM.workspace_id,
                        NoteSimilarityORM.note_chunk_id,
                    ).in_(note_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(NoteSimilarityORM), rows)
                session.commit()

                log.info(
                    "workspace.bulk_insert_notes_similarities.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_notes_similarities.error_upsert",
                    error=str(e),
                )
                raise e

    def get_chunks_by_note_id(self, note_id: str) -> List[NoteChunkORM]:
        try:
            note_uuid: UUID = uuid.UUID(note_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {note_id}")

        try:
            with self._db_pool.session() as session:
                stmt = select(NoteChunkORM).where(NoteChunkORM.note_id == note_uuid)
                result = session.execute(stmt).scalars().all()
                return list(result)

        except SQLAlchemyError as e:
            log.error(
                "workspace_note.error_get_chunks_by_note_id",
                note_id=note_id,
                error=str(e),
            )
            raise e

    def match_note_with_papers(
        self,
        chunks: List[NoteChunkORM],
        top_k: int = 15,
        candidate_pool_size: Optional[int] = None,
    ) -> Dict[Any, MatchResultDict]:
        """Mencari paper yang paling relevan untuk setiap chunk dari note.

        Menggunakan Opsi B (Weighted Score):
        - Basis utama: Kualitas chunk terbaik (MIN distance / MAX similarity).
        - Bonus kepadatan: Dibagi dengan LN(matched_chunk_count + 1) untuk memberikan
        peringkat lebih baik bagi paper dengan banyak chunk relevan tanpa
        menciptakan long-paper bias secara ekstrem.
        """
        results: Dict[Any, MatchResultDict] = {}

        try:
            with self._db_pool.session() as session:
                for chunk in chunks:
                    if not chunk.embedding:
                        log.warning(
                            "workspace_note.empty_embedding",
                            chunk_id=getattr(chunk, "id", None),
                        )
                        results[chunk.id] = {
                            "id": chunk.id,
                            "chunk_content": chunk.content,
                            "papers": [],
                            "similar_chunks": [],
                        }
                        continue

                    # Buffer kandidat awal untuk pencarian pgvector HNSW/IVFFlat index.
                    # Disarankan minimal top_k * 10 agar ada cukup sampel multi-chunk per paper.
                    pool_size = candidate_pool_size or (top_k * 10)

                    # Expression pgvector cosine distance (1 - cosine similarity)
                    distance_expr = DocumentChunkORM.embedding.cosine_distance(
                        chunk.embedding
                    )

                    # -------------------------------------------------------------
                    # 1. CTE: Ambil kandidat chunk terdekat (Memaksa penggunaan Vector Index)
                    # -------------------------------------------------------------
                    top_candidates = (
                        select(
                            DocumentChunkORM.id.label("document_chunk_id"),
                            DocumentChunkORM.paper_id.label("paper_id"),
                            DocumentChunkORM.content.label(
                                "document_chunk_content"
                            ),
                            distance_expr.label("distance"),
                        )
                        .order_by(distance_expr.asc())
                        .limit(pool_size)
                        .cte("top_candidates")
                    )

                    # -------------------------------------------------------------
                    # 2. CTE: Agregasi per paper & Hitung Weighted Score (Opsi B)
                    #    Formula: weighted_score = min_distance / LN(count + 1.0)
                    #    (Semakin KECIL weighted_score, semakin RELEVAN paper tersebut)
                    # -------------------------------------------------------------
                    paper_metrics = (
                        select(
                            top_candidates.c.paper_id,
                            func.min(top_candidates.c.distance).label(
                                "min_distance"
                            ),
                            func.count(top_candidates.c.document_chunk_id).label(
                                "matched_chunk_count"
                            ),
                            (
                                func.min(top_candidates.c.distance)
                                / func.ln(
                                    func.count(top_candidates.c.document_chunk_id)
                                    + 1.0
                                )
                            ).label("weighted_score"),
                        )
                        .group_by(top_candidates.c.paper_id)
                        .cte("paper_metrics")
                    )

                    # -------------------------------------------------------------
                    # 3. CTE: Ranking Paper berdasarkan weighted_score
                    # -------------------------------------------------------------
                    ranked_papers = (
                        select(
                            paper_metrics.c.paper_id,
                            paper_metrics.c.min_distance,
                            paper_metrics.c.matched_chunk_count,
                            paper_metrics.c.weighted_score,
                            func.row_number()
                            .over(order_by=paper_metrics.c.weighted_score.asc())
                            .label("paper_rank"),
                        )
                        .cte("ranked_papers")
                    )

                    # Filter hanya top_k paper terbaik
                    top_papers = (
                        select(
                            ranked_papers.c.paper_id,
                            ranked_papers.c.min_distance,
                            ranked_papers.c.matched_chunk_count,
                            ranked_papers.c.weighted_score,
                        )
                        .where(ranked_papers.c.paper_rank <= top_k)
                        .cte("top_papers")
                    )

                    # -------------------------------------------------------------
                    # 4. QUERY UTAMA: Join Detail PaperORM + SEMUA Chunk Relevan
                    # -------------------------------------------------------------
                    stmt = (
                        select(
                            PaperORM,
                            top_candidates.c.document_chunk_id,
                            top_candidates.c.document_chunk_content,
                            top_candidates.c.distance,
                            top_papers.c.min_distance,
                            top_papers.c.matched_chunk_count,
                            top_papers.c.weighted_score,
                        )
                        .join(top_papers, top_papers.c.paper_id == PaperORM.id)
                        .join(
                            top_candidates,
                            top_candidates.c.paper_id == PaperORM.id,
                        )
                        .order_by(
                            top_papers.c.weighted_score.asc(),
                            top_candidates.c.distance.asc(),
                        )
                    )

                    rows = session.execute(stmt).all()

                    # -------------------------------------------------------------
                    # 5. Formatting Hasil & Menghindari DetachedInstanceError
                    # -------------------------------------------------------------
                    papers_by_id: Dict[Any, Dict[str, Any]] = {}
                    paper_order: List[Any] = []
                    similar_chunks: List[SimilarChunkDict] = []

                    for (
                        paper_orm,
                        document_chunk_id,
                        document_chunk_content,
                        distance,
                        min_distance,
                        matched_chunk_count,
                        weighted_score,
                    ) in rows:
                        if paper_orm.id not in papers_by_id:
                            # Ekstrak atribut ORM ke dictionary sebelum session ditutup
                            paper_dict = {
                                column.name: getattr(paper_orm, column.name)
                                for column in paper_orm.__table__.columns
                            }

                            # Tambahkan metrik kustom ke objek paper
                            paper_dict["best_similarity_score"] = float(
                                1 - min_distance
                            )
                            paper_dict["matched_chunk_count"] = matched_chunk_count
                            paper_dict["weighted_score"] = float(weighted_score)

                            papers_by_id[paper_orm.id] = paper_dict
                            paper_order.append(paper_orm.id)

                        similar_chunks.append(
                            {
                                "document_content": document_chunk_content,
                                "document_id": document_chunk_id,
                                "chunk_content": chunk.content,
                                "chunk_id": chunk.id,
                                "paper_id": paper_orm.id,
                                "similarity_score": float(1 - distance),
                            }
                        )

                    papers = [papers_by_id[pid] for pid in paper_order]

                    results[chunk.id] = {
                        "id": chunk.id,
                        "chunk_content": chunk.content,
                        "papers": papers,
                        "similar_chunks": similar_chunks,
                    }

        except SQLAlchemyError:
            log.exception("workspace_note.error_match_note_with_papers")
            raise

        log.info("workspace_note.match_note_with_papers", chunks=len(results))

        return results
