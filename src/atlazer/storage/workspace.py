import uuid
import structlog

from uuid import UUID
from atlazer.models.paper import PaperORM

from datetime import datetime
from typing import List, Any, Dict
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import ContextChunkORM, ContextPaperORM, ContextSimilarityORM
from atlazer.models.document import DocumentChunkORM

from sqlalchemy import tuple_, insert, select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = structlog.get_logger()


class WorkspaceDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def bulk_insert_chunks(self, values: List[ContextChunkORM]) -> None:
        if not values:
            log.info("workspace.bulk_insert_chunks.empty_list")
            return

        user_pairs = list({(chunk.user_id, chunk.context_id, chunk.workspace_id, chunk.chunk_index) for chunk in values})
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
                        ContextChunkORM.chunk_index,
                    ).in_(user_pairs)
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
        chunks: List[ContextChunkORM],
        top_k: int = 15,
    ) -> Dict[str, Any]:
        result: Dict[Any, Any] = {}

        try:
            with self._db_pool.session() as session:
                for chunk in chunks:
                    if chunk.embedding is None or len(chunk.embedding) == 0:
                        log.warning("context_chunk.empty_embedding", chunk=chunk)
                        result[chunk.id] = {
                            "id": chunk.id,
                            "chunk_content": chunk.content,
                            "papers": [],
                            "similar_chunks": [],
                        }
                        continue

                    distance = DocumentChunkORM.embedding.cosine_distance(chunk.embedding)

                    # ambil 1 document-chunk terdekat per paper untuk chunk ini
                    ranked_subq = (
                        select(
                            DocumentChunkORM.id.label("document_chunk_id"),
                            DocumentChunkORM.content.label("document_chunk_content"),
                            DocumentChunkORM.paper_id.label("paper_id"),
                            distance.label("distance"),
                            func.row_number()
                            .over(
                                partition_by=DocumentChunkORM.paper_id,
                                order_by=distance.asc(),
                            )
                            .label("rn"),
                        )
                        .subquery()
                    )

                    stmt = (
                        select(
                            PaperORM,
                            ranked_subq.c.document_chunk_id,
                            ranked_subq.c.document_chunk_content,
                            ranked_subq.c.distance,
                        )
                        .join(ranked_subq, ranked_subq.c.paper_id == PaperORM.id)
                        .where(ranked_subq.c.rn == 1)
                        .order_by(ranked_subq.c.distance.asc())
                        .limit(top_k)
                    )

                    rows = session.execute(stmt).all()

                    papers: List[PaperORM] = []
                    similar_chunks: List[dict] = []

                    for paper, doc_chunk_id, doc_chunk_content, dist in rows:
                        papers.append(paper)
                        similar_chunks.append(
                            {
                                "document_content": doc_chunk_content,
                                "document_id": doc_chunk_id,
                                "chunk_content": chunk.content,
                                "chunk_id": chunk.id,
                                "similarity_score": round(1 - dist, 4),
                                "paper_id": paper.id,
                            }
                        )

                    result[chunk.id] = {
                        "id": chunk.id,
                        "chunk_content": chunk.content,
                        "papers": papers,
                        "similar_chunks": similar_chunks,
                    }

        except SQLAlchemyError as e:
            log.error(
                "context_chunk.error_match_context_with_papers",
                error=str(e),
            )
            raise e

        log.info(
            "context_chunk.match_context_with_papers",
            chunks=len(result),
        )

        return result
