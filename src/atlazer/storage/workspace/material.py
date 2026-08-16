import uuid
import structlog

from typing import List, Optional, Any
from datetime import date
from sqlalchemy import update, tuple_, insert
from sqlalchemy.exc import SQLAlchemyError
from atlazer.models.workspace import (
    LearningMaterialDocumentORM,
    LearningMaterialNoteORM,
    LearningMaterialSourceORM,
    LearningMaterialORM,
)
from atlazer.storage.db import DatabasePool

log = structlog.get_logger()


class LearningMaterialDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def get_documents(
        self,
        workspace_id: str,
        processing_date: str
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
            clustered_date_obj = date.fromisoformat(processing_date)
        except ValueError:
            log.error(
                "workspace.get_learning_material_documents.error_date",
                clustered_date=processing_date,
            )
            raise ValueError(f"Invalid clustered_date: {processing_date}")

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

    def update_documents_with_label(
        self,
        values: List[LearningMaterialDocumentORM]
    ) -> List[LearningMaterialDocumentORM]:
        if not values:
            log.info("workspace_document.update_documents_with_label.empty_list")
            return []

        rows = [
            {
                "id": doc.id,
                "attributes": doc.attributes,
                "cluster_label": doc.cluster_label,
                "clustered_date": doc.clustered_date,
            }
            for doc in values
            if doc.id is not None
        ]

        if not rows:
            return values

        with self._db_pool.session() as session:
            try:
                session.execute(update(LearningMaterialDocumentORM), rows)
                session.commit()

                log.info(
                    "workspace_document.update_documents_with_label.finish",
                    count=len(rows),
                )
                return values

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace_document.update_documents_with_label.error",
                    error=str(e),
                )
                raise e

    def insert_sources(self, values: List[LearningMaterialSourceORM]) -> None:
        if not values:
            log.info("workspace.material.insert_sources.empty_list")
            return

        source_keys = list({
            (chunk.clustered_date, chunk.cluster_label, chunk.workspace_id) 
            for chunk in values
        })

        rows = [
            {
                "workspace_id": chunk.workspace_id,
                "cluster_label": chunk.cluster_label,
                "clustered_date": chunk.clustered_date,
                "chunks": chunk.chunks,
                "content": chunk.content,
                "attributes": chunk.attributes,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                session.query(LearningMaterialSourceORM).filter(
                    tuple_(
                        LearningMaterialSourceORM.clustered_date, 
                        LearningMaterialSourceORM.cluster_label,
                        LearningMaterialSourceORM.workspace_id,
                    ).in_(source_keys)
                ).delete(synchronize_session=False)

                session.execute(insert(LearningMaterialSourceORM), rows)
                session.commit()

                log.info(
                    "workspace.material.insert_sources.finish_upsert",
                    count=len(values),
                )

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.material.insert_sources.error_upsert",
                    error=str(e),
                )
                raise e

    def get_sources(
        self,
        workspace_id: str,
        processing_date: str
    ) -> List[LearningMaterialSourceORM]:
        try:
            workspace_uuid = uuid.UUID(workspace_id)
        except ValueError:
            log.error(
                "workspace.get_learning_material_sources.error_uuid",
                workspace_id=workspace_id,
            )
            raise ValueError(f"Invalid workspace_id: {workspace_id}")

        try:
            clustered_date_obj = date.fromisoformat(processing_date)
        except ValueError:
            log.error(
                "workspace.get_learning_material_sources.error_date",
                clustered_date=processing_date,
            )
            raise ValueError(f"Invalid clustered_date: {processing_date}")

        with self._db_pool.session() as session:
            try:
                return (
                    session.query(LearningMaterialSourceORM)
                    .filter(
                        LearningMaterialSourceORM.workspace_id == workspace_uuid,
                        LearningMaterialSourceORM.clustered_date == clustered_date_obj,
                    )
                    .all()
                )
            except SQLAlchemyError as e:
                log.error(
                    "workspace.get_learning_material_sources.error_select",
                    error=str(e),
                )
                raise e

    def insert_material(self, value: LearningMaterialORM) -> Optional[Any]:
        if not value:
            log.info("workspace.material.insert_material.empty_value")
            return None

        row = {
            "workspace_id": value.workspace_id,
            "content": value.content,
            "attributes": value.attributes,
            "generated_date": value.generated_date,
        }

        with self._db_pool.session() as session:
            try:
                session.query(LearningMaterialORM).filter(
                    LearningMaterialORM.workspace_id == value.workspace_id
                ).delete(synchronize_session=False)

                result = session.execute(
                    insert(LearningMaterialORM).returning(LearningMaterialORM.id),
                    row,
                )
                material_id = result.scalar_one()
                session.commit()

                log.info(
                    "workspace.material.insert_material.finish_upsert",
                    workspace_id=value.workspace_id,
                    material_id=material_id,
                )

                return material_id

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "workspace.material.insert_material.error_upsert",
                    error=str(e),
                )
                raise e


    def get_material_by_id(self, material_id: Any) -> Optional[LearningMaterialORM]:
        with self._db_pool.session() as session:
            try:
                material = (
                    session.query(LearningMaterialORM)
                    .filter(LearningMaterialORM.id == material_id)
                    .one_or_none()
                )

                if material is None:
                    log.info(
                        "workspace.material.get_material_by_id.not_found",
                        material_id=material_id,
                    )

                return material

            except SQLAlchemyError as e:
                log.error(
                    "workspace.material.get_material_by_id.error",
                    material_id=material_id,
                    error=str(e),
                )
                raise e
