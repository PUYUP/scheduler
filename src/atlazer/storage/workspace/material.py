import uuid
import structlog

from typing import List
from datetime import date
from sqlalchemy.exc import SQLAlchemyError
from atlazer.models.workspace import (
    LearningMaterialDocumentORM,
    LearningMaterialNoteORM,
)
from atlazer.storage.db import DatabasePool

log = structlog.get_logger()


class LearningMaterialDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def get_documents(
        self,
        workspace_id: str,
        clustered_date: str
    ) -> List[LearningMaterialDocumentORM]:
        try:
            workspace_uuid = uuid.UUID(workspace_id)
        except ValueError:
            log.error(
                "workspace.get_learning_material_documents.error_uuid",
                workspace_id=workspace_id,
            )
            raise ValueError(f"Invalid workspace_id: {workspace_id}")

        try:
            clustered_date_obj = date.fromisoformat(clustered_date)
        except ValueError:
            log.error(
                "workspace.get_learning_material_documents.error_date",
                clustered_date=clustered_date,
            )
            raise ValueError(f"Invalid clustered_date: {clustered_date}")

        with self._db_pool.session() as session:
            try:
                return (
                    session.query(LearningMaterialDocumentORM)
                    .join(
                        LearningMaterialNoteORM,
                        LearningMaterialNoteORM.id == LearningMaterialDocumentORM.material_note_id,
                    )
                    .filter(
                        LearningMaterialDocumentORM.workspace_id == workspace_uuid,
                        LearningMaterialNoteORM.clustered_date == clustered_date_obj,
                    )
                    .all()
                )
            except SQLAlchemyError as e:
                log.error(
                    "workspace.get_learning_material_documents.error_select",
                    error=str(e),
                )
                raise e
