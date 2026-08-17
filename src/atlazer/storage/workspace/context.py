import uuid
import structlog

from uuid import UUID
from atlazer.models.paper import PaperORM

from collections import defaultdict
from typing import List, Any, Dict, TypedDict, Optional
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import (
    ContextChunkORM,
    ContextPaperORM,
    ContextDocumentORM,
    ContextPaperSummaryORM
)
from atlazer.models.document import DocumentChunkORM

from sqlalchemy import tuple_, insert, select, func
from sqlalchemy.exc import SQLAlchemyError

log = structlog.get_logger()


# --- 1. Type Hinting Setup ---
class SimilarChunkDict(TypedDict):
    document_content: str
    document_embedding: List[float]
    document_id: Any
    chunk_content: str
    chunk_id: Any
    paper_id: Any
    similarity_score: float


class MatchResultDict(TypedDict):
    id: Any
    chunk_content: str
    chunk_embedding: List[float]
    papers: List[Dict[str, Any]]  # Berisi list of dictionary dari paper
    similar_chunks: List[SimilarChunkDict]


class WorkspaceContextDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def insert_context_chunks(self, values: List[ContextChunkORM]) -> None:
        if not values:
            log.info("workspace.insert_context_chunks.empty_list")
            return

        context_keys = list({
            (chunk.user_id, chunk.context_id, chunk.workspace_id) 
            for chunk in values
        })

        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "context_id": chunk.context_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "attributes": chunk.attributes,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(ContextChunkORM).filter(
                    tuple_(
                        ContextChunkORM.user_id, 
                        ContextChunkORM.context_id,
                        ContextChunkORM.workspace_id,
                    ).in_(context_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextChunkORM), rows)
                session.commit()

                log.info(
                    "workspace.insert_context_chunks.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.insert_context_chunks.error_upsert",
                    error=str(e),
                )
                raise e

    def insert_context_papers(self, values: List[ContextPaperORM]) -> None:
        if not values:
            log.info("workspace.insert_context_papers.empty_list")
            return

        context_keys = list({(chunk.user_id, chunk.context_id) for chunk in values})
        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "context_id": chunk.context_id,
                "paper_id": chunk.paper_id,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(ContextPaperORM).filter(
                    tuple_(
                        ContextPaperORM.user_id, 
                        ContextPaperORM.context_id
                    ).in_(context_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextPaperORM), rows)
                session.commit()

                log.info(
                    "workspace.insert_context_papers.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.insert_context_papers.error_upsert",
                    error=str(e),
                )
                raise e

    def insert_context_documents(
        self,
        values: List[ContextDocumentORM]
    ) -> None:
        if not values:
            log.info("workspace.insert_context_documents.empty_list")
            return

        context_keys = list({(chunk.user_id, chunk.context_id, chunk.workspace_id, chunk.context_chunk_id) for chunk in values})
        rows = [
            {
                "user_id": chunk.user_id,
                "workspace_id": chunk.workspace_id,
                "context_id": chunk.context_id,
                "context_chunk_id": chunk.context_chunk_id,
                "context_content": chunk.context_content,
                "paper_id": chunk.paper_id,
                "document_chunk_id": chunk.document_chunk_id,
                "document_content": chunk.document_content,
                "document_embedding": chunk.document_embedding,
                "attributes": chunk.attributes,
                "similarity_score": chunk.similarity_score,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(ContextDocumentORM).filter(
                    tuple_(
                        ContextDocumentORM.user_id, 
                        ContextDocumentORM.context_id,
                        ContextDocumentORM.workspace_id,
                        ContextDocumentORM.context_chunk_id,
                    ).in_(context_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextDocumentORM), rows)
                session.commit()

                log.info(
                    "workspace.insert_context_documents.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.insert_context_documents.error_upsert",
                    error=str(e),
                )
                raise e

    def insert_paper_summaries(
        self,
        values: List[ContextPaperSummaryORM]
    ) -> None:
        if not values:
            log.info("workspace.insert_paper_summaries.empty_list")
            return

        context_keys = list({(chunk.context_id, chunk.workspace_id) for chunk in values})
        rows = [
            {
                "workspace_id": chunk.workspace_id,
                "context_id": chunk.context_id,
                "content": chunk.content,
                "paper_id": chunk.paper_id,
                "attributes": chunk.attributes,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(ContextPaperSummaryORM).filter(
                    tuple_(
                        ContextPaperSummaryORM.context_id,
                        ContextPaperSummaryORM.workspace_id,
                    ).in_(context_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextPaperSummaryORM), rows)
                session.commit()

                log.info(
                    "workspace.insert_paper_summaries.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.insert_paper_summaries.error_upsert",
                    error=str(e),
                )
                raise e

    def get_chunks_by_context_id(
        self,
        context_id: str
    ) -> List[ContextChunkORM]:
        try:
            context_uuid: UUID = uuid.UUID(context_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {context_id}")

        try:
            with self._db_pool.session() as session:
                stmt = select(ContextChunkORM).where(ContextChunkORM.context_id == context_uuid)
                result = session.execute(stmt).scalars().all()
                return list(result)

        except SQLAlchemyError as e:
            log.error(
                "context_chunk.error_get_chunks_by_context_id",
                context_id=context_id,
                error=str(e),
            )
            raise e

    def get_documents_by_context_id(
        self, 
        context_id: str
    ) -> Dict[Any, List[ContextDocumentORM]]:
        try:
            context_uuid: uuid.UUID = uuid.UUID(context_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {context_id}")

        try:
            with self._db_pool.session() as session:
                stmt = select(ContextDocumentORM).where(ContextDocumentORM.context_id == context_uuid)
                result = session.execute(stmt).scalars().all()
                
                grouped_result = defaultdict(list)
                for item in result:
                    grouped_result[item.paper_id].append(item)
                    
                return dict(grouped_result)

        except SQLAlchemyError as e:
            log.error(
                "workspace_context.error_get_documents_by_context_id",
                context_id=context_id,
                error=str(e),
            )
            raise e

    def match_context_with_papers(
        self,
        chunks: List[ContextChunkORM],
        top_k: int = 15,
        candidate_pool_size: Optional[int] = None,
        min_similarity: float = 0.0,
    ) -> Dict[Any, MatchResultDict]:
        """Mencari paper yang paling relevan untuk setiap chunk dari context.

        Menggunakan Opsi B (Weighted Score):
        - Basis utama: Kualitas chunk terbaik (MIN distance / MAX similarity).
        - Bonus kepadatan: Dibagi dengan LN(matched_chunk_count + 1) untuk memberikan
        peringkat lebih baik bagi paper dengan banyak chunk relevan tanpa
        menciptakan long-paper bias secara ekstrem.
        - Filter: Hanya menyertakan chunk dengan similarity >= min_similarity.
        """
        results: Dict[Any, MatchResultDict] = {}

        try:
            with self._db_pool.session() as session:
                for chunk in chunks:
                    if chunk.embedding is None or len(chunk.embedding) == 0:
                        log.warning(
                            "workspace_context.empty_embedding",
                            chunk_id=getattr(chunk, "id", None),
                        )
                        results[chunk.id] = {
                            "id": chunk.id,
                            "chunk_content": chunk.content,
                            "chunk_embedding": chunk.embedding if chunk.embedding is not None else [],
                            "papers": [],
                            "similar_chunks": [],
                        }
                        continue

                    # Buffer kandidat awal untuk pencarian pgvector HNSW/IVFFlat index.
                    pool_size = candidate_pool_size or (top_k * 10)

                    # Expression pgvector cosine distance (1 - cosine similarity)
                    distance_expr = DocumentChunkORM.embedding.cosine_distance(
                        chunk.embedding
                    )

                    # -------------------------------------------------------------
                    # 1. CTE: Ambil kandidat chunk terdekat
                    # -------------------------------------------------------------
                    # Base query untuk top candidates
                    top_candidates_query = select(
                        DocumentChunkORM.id.label("document_chunk_id"),
                        DocumentChunkORM.paper_id.label("paper_id"),
                        DocumentChunkORM.content.label("document_chunk_content"),
                        DocumentChunkORM.embedding.label("document_chunk_embedding"),
                        distance_expr.label("distance"),
                    )

                    # Terapkan filter berdasarkan argumen min_similarity
                    if min_similarity > -1.0:
                        max_distance = 1.0 - min_similarity
                        top_candidates_query = top_candidates_query.where(
                            distance_expr <= max_distance
                        )

                    top_candidates = (
                        top_candidates_query.order_by(distance_expr.asc())
                        .limit(pool_size)
                        .cte("top_candidates")
                    )

                    # -------------------------------------------------------------
                    # 2. CTE: Agregasi per paper & Hitung Weighted Score (Opsi B)
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
                            top_candidates.c.document_chunk_embedding,
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
                        document_chunk_embedding,
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
                                "document_embedding": document_chunk_embedding,
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
                        "chunk_embedding": chunk.embedding if chunk.embedding is not None else [],
                        "papers": papers,
                        "similar_chunks": similar_chunks,
                    }

        except SQLAlchemyError:
            log.exception("workspace_context.error_match_context_with_papers")
            raise

        log.info("workspace_context.match_context_with_papers", chunks=len(results))

        return results
