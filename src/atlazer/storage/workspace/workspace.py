import uuid
import logging

from datetime import datetime, timezone
from typing import List
from uuid import UUID
from atlazer.storage.db import DatabasePool
from atlazer.models.workspace import WorkspaceORM
from sqlalchemy import update, select, or_
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class WorkspaceDepot:

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    def _values(self, payload: WorkspaceORM) -> dict:
        return {
            "next_notes_processing_at": payload.next_notes_processing_at
        }

    def update(self, uuid_str: str, payload: WorkspaceORM) -> None:
        values = self._values(payload)
        if not values:
            return

        try:
            workspace_uuid: UUID = uuid.UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {uuid_str}")

        stmt = (
            update(WorkspaceORM)
                .where(WorkspaceORM.id == workspace_uuid)
                .values(**values)
                .returning(WorkspaceORM.id)
                .execution_options(synchronize_session="fetch")
        )

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)
                updated_id = result.scalar()

                if updated_id is None:
                    session.rollback()
                    raise ValueError(
                        f"Could not update workspace "
                        f"id={workspace_uuid!r} "
                        "(no matching row)"
                    )

                session.commit()
        except SQLAlchemyError:
            logger.exception("Failed to update workspace id=%s", workspace_uuid)
            raise

    def update_bulk(self, payloads: list[WorkspaceORM]) -> None:
        if not payloads:
            return

        rows = []
        for p in payloads:
            if p.id is None:
                continue
            row = self._values(p)
            row["id"] = p.id
            rows.append(row)

        if not rows:
            return

        try:
            with self._pool.session() as session:
                session.execute(update(WorkspaceORM), rows)
                session.commit()
        except SQLAlchemyError:
            logger.exception("Failed to bulk update workspaces")
            raise

    def get_pre_processing_workspaces(self) -> List[WorkspaceORM]:
        """Get workspaces that are ready for notes processing."""
        current_time = datetime.now(timezone.utc).isoformat()

        # Gunakan .is_(None) yang merupakan standar SQLAlchemy untuk perbandingan NULL
        stmt = (
            select(WorkspaceORM)
            .where(
                or_(
                    WorkspaceORM.next_notes_processing_at.is_(None),
                    WorkspaceORM.next_notes_processing_at < current_time,
                )
            )
            .limit(10)
        )

        with self._pool.session() as session:
            try:
                rows = session.execute(stmt).scalars().all()
                return list(rows)
            except Exception as e:
                # Menggunakan logger.exception agar merekam stack-trace penuh
                logger.exception("workspace.get_pre_processing_workspaces.failed: %s", str(e))
                raise
