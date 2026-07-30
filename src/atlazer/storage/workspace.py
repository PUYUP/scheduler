import uuid
import structlog

from uuid import UUID
from atlazer.models.paper import PaperORM

from datetime import datetime
from typing import List, Any, Dict
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import ContextChunkORM, ContextPaperORM
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
        top_k: int = 10,
    ) -> Dict[str, Any]:
        paper_distances: Dict[Any, List[float]] = {}
        paper_objects: Dict[Any, PaperORM] = {}
        raw_matches: List[dict] = []

        try:
            with self._db_pool.session() as session:
                for chunk in chunks:
                    if not chunk.embedding or len(chunk.embedding) == 0:
                        log.warning("context_chunk.empty_embedding", chunk=chunk)
                        continue

                    distance = DocumentChunkORM.embedding.cosine_distance(chunk.embedding)

                    # ambil 1 document-chunk terdekat per paper untuk chunk ini
                    ranked_subq = (
                        select(
                            DocumentChunkORM.id.label("document_chunk_id"),
                            DocumentChunkORM.content.label("document_chunk_content"),
                            DocumentChunkORM.paper_id.label("paper_id"),
                            distance.label("distance"),
                            func
                                .row_number()
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

                    for paper, doc_chunk_id, doc_chunk_content, dist in rows:
                        paper_distances.setdefault(paper.id, []).append(dist)
                        paper_objects.setdefault(paper.id, paper)

                        raw_matches.append(
                            {
                                "dockument_content": doc_chunk_content,
                                "document_id": doc_chunk_id,
                                "chunk_content": chunk.content,
                                "chunk_id": chunk.id,
                                "similarity_score": round(1 - dist, 4),
                                "paper_id": paper.id,
                            }
                        )

        except SQLAlchemyError as e:
            log.error(
                "context_chunk.error_match_context_with_papers",
                error=str(e),
            )
            raise e

        # rata-ratakan distance tiap paper lintas semua chunk yang match ke paper itu
        averaged = [
            (paper_objects[pid], sum(dists) / len(dists))
            for pid, dists in paper_distances.items()
        ]
        averaged.sort(key=lambda item: item[1])

        top_papers: List[PaperORM] = [paper for paper, _ in averaged[:top_k]]
        top_paper_ids = {paper.id for paper in top_papers}

        similar_chunks = [
            match for match in raw_matches if match["paper_id"] in top_paper_ids
        ]

        result: Dict[str, Any] = {
            "papers": top_papers,
            "similar_chunks": similar_chunks,
        }

        log.info(
            "context_chunk.match_context_with_papers",
            papers=len(top_papers),
            similar_chunks=len(similar_chunks),
        )

        return result
