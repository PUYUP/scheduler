import structlog
import uuid

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Any
from uuid import UUID

from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import (
    ContextChunkORM,
)

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert, select, tuple_

log = structlog.get_logger()


class ContextChunkDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def bulk_upsert(self, values: List[ContextChunkORM]) -> None:
        if not values:
            log.info("context_chunk.empty_list")
            return

        # combination for deletion
        user_pairs = list({(chunk.user_id, chunk.workspace_id, chunk.context_id) for chunk in values})

        # rows for upsert
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
                # 3. DELETE old chunks
                session.query(ContextChunkORM).filter(
                    tuple_(
                        ContextChunkORM.user_id, 
                        ContextChunkORM.workspace_id,
                        ContextChunkORM.context_id
                    ).in_(user_pairs)
                ).delete(synchronize_session=False)

                # 4. INSERT new chunks
                session.execute(insert(ContextChunkORM), rows)
                session.commit()
                
                log.info(
                    "context_chunk.finish_reindex",
                    user_pairs=str(user_pairs),
                    count=len(values)
                )
                
            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "context_chunk.error_reindex",
                    error=str(e),
                )
                raise e
