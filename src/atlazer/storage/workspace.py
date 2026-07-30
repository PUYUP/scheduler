import uuid
import structlog

from uuid import UUID
from atlazer.models.paper import PaperORM

from typing import List, Any, Dict, TypedDict
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import ContextChunkORM, ContextPaperORM, ContextSimilarityORM
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


class WorkspaceDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def bulk_insert_chunks(self, values: List[ContextChunkORM]) -> None:
        if not values:
            log.info("workspace.bulk_insert_chunks.empty_list")
            return

        context_keys = list({
            (chunk.user_id, chunk.workspace_id, chunk.context_id) 
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
                    "workspace.bulk_insert_chunks.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_chunks.error_upsert",
                    error=str(e),
                )
                raise e

    def bulk_insert_papers(self, values: List[ContextPaperORM]) -> None:
        if not values:
            log.info("workspace.bulk_insert_chunks.empty_list")
            return

        user_pairs = list({(chunk.user_id, chunk.context_id) for chunk in values})
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
                    ).in_(user_pairs)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextPaperORM), rows)
                session.commit()

                log.info(
                    "workspace.bulk_insert_papers.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_papers.error_upsert",
                    error=str(e),
                )
                raise e

    def bulk_insert_similarities(
        self,
        values: List[ContextSimilarityORM]
    ) -> None:
        if not values:
            log.info("workspace.bulk_insert_similarities.empty_list")
            return

        user_pairs = list({(chunk.user_id, chunk.context_id, chunk.workspace_id, chunk.context_chunk_id) for chunk in values})
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
                "attributes": chunk.attributes,
                "similarity_score": chunk.similarity_score,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(ContextSimilarityORM).filter(
                    tuple_(
                        ContextSimilarityORM.user_id, 
                        ContextSimilarityORM.context_id,
                        ContextSimilarityORM.workspace_id,
                        ContextSimilarityORM.context_chunk_id,
                    ).in_(user_pairs)
                ).delete(synchronize_session=False)

                session.execute(insert(ContextSimilarityORM), rows)
                session.commit()

                log.info(
                    "workspace.bulk_insert_similarities.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.bulk_insert_similarities.error_upsert",
                    error=str(e),
                )
                raise e

    def get_chunks_by_context_id(self, context_id: str) -> List[ContextChunkORM]:
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

    def match_context_with_papers(
        self,
        chunks: List['ContextChunkORM'], # Sesuaikan tipe kelas Anda
        top_k: int = 15,
    ) -> Dict[Any, MatchResultDict]:

        results: Dict[Any, MatchResultDict] = {}

        try:
            with self._db_pool.session() as session:
                for chunk in chunks:
                    if not chunk.embedding:
                        log.warning(
                            "context_chunk.empty_embedding",
                            chunk=chunk,
                        )

                        results[chunk.id] = {
                            "id": chunk.id,
                            "chunk_content": chunk.content,
                            "papers": [],
                            "similar_chunks": [],
                        }
                        continue

                    # Jarak cosine
                    distance_expr = (
                        DocumentChunkORM.embedding.cosine_distance(chunk.embedding)
                    )

                    # --- 2. Perbaikan Query (Memaksa penggunaan Vector Index) ---
                    # Ambil kandidat awal dengan LIMIT agar pgvector (atau ekstensilain) menggunakan Index.
                    # Kita buffer (top_k * 5) karena beberapa chunk mungkin berasal dari paper yang sama.
                    top_candidates = (
                        select(
                            DocumentChunkORM.id.label("document_chunk_id"),
                            DocumentChunkORM.paper_id.label("paper_id"),
                            DocumentChunkORM.content.label("document_chunk_content"),
                            distance_expr.label("distance"),
                        )
                        .order_by(distance_expr.asc())
                        .limit(top_k * 5) 
                        .cte("top_candidates")
                    )

                    # Lakukan pemeringkatan (window function) dari kandidat yang sudah di-limit
                    ranked_chunks = (
                        select(
                            top_candidates.c.document_chunk_id,
                            top_candidates.c.paper_id,
                            top_candidates.c.document_chunk_content,
                            top_candidates.c.distance,
                            func.row_number()
                            .over(
                                partition_by=top_candidates.c.paper_id,
                                order_by=top_candidates.c.distance.asc(),
                            )
                            .label("rank"),
                        )
                        .cte("ranked_chunks")
                    )

                    # Ambil paper unik terbaik
                    stmt = (
                        select(
                            PaperORM,
                            ranked_chunks.c.document_chunk_id,
                            ranked_chunks.c.document_chunk_content,
                            ranked_chunks.c.distance,
                        )
                        .join(
                            ranked_chunks,
                            ranked_chunks.c.paper_id == PaperORM.id,
                        )
                        .where(ranked_chunks.c.rank == 1)
                        .order_by(ranked_chunks.c.distance.asc())
                        .limit(top_k)
                    )

                    rows = session.execute(stmt).all()
                    
                    papers = []
                    similar_chunks: List[SimilarChunkDict] = []

                    for (
                        paper_orm,
                        document_chunk_id,
                        document_chunk_content,
                        distance,
                    ) in rows:
                        
                        # --- 3. Mencegah DetachedInstanceError ---
                        # Ubah ORM object menjadi dictionary biasa sebelum session ditutup
                        # Anda bisa memanggil spesifik atribut (contoh: id, title, abstract)
                        # atau menggunakan pendekatan dinamis seperti di bawah ini:
                        paper_dict = {
                            column.name: getattr(paper_orm, column.name)
                            for column in paper_orm.__table__.columns
                        }
                        
                        papers.append(paper_dict)
                        
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

                    results[chunk.id] = {
                        "id": chunk.id,
                        "chunk_content": chunk.content,
                        "papers": papers,
                        "similar_chunks": similar_chunks,
                    }

        except SQLAlchemyError:
            log.exception(
                "context_chunk.error_match_context_with_papers",
            )
            raise

        log.info(
            "context_chunk.match_context_with_papers",
            chunks=len(results),
        )

        return results
