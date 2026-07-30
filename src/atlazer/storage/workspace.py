import structlog

from datetime import datetime
from typing import List
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import (
    ContextChunkORM,
)

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = structlog.get_logger()


class ContextChunkDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def bulk_upsert(self, values: List[ContextChunkORM]) -> None:
        if not values:
            log.info("context_chunk.empty_list")
            return

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
                stmt = pg_insert(ContextChunkORM).values(rows)

                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        ContextChunkORM.user_id,
                        ContextChunkORM.workspace_id,
                        ContextChunkORM.context_id,
                        ContextChunkORM.chunk_index,
                    ],
                    set_={
                        "content": stmt.excluded.content,
                        "embedding": stmt.excluded.embedding,
                        "attributes": stmt.excluded.attributes,
                        "updated_at": datetime.now(),
                    },
                )

                session.execute(stmt)
                session.commit()

                log.info(
                    "context_chunk.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "context_chunk.error_upsert",
                    error=str(e),
                )
                raise e
