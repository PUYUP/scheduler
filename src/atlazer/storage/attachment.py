import structlog
from typing import Optional, Any
from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError

from atlazer.storage.db import DatabasePool
from atlazer.models.attachment import FileORM, AttachmentORM


log = structlog.get_logger()


class AttachmentDepot:
    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def insert_file(self, value: FileORM) -> Optional[Any]:
        if not value:
            log.info("storage.attachment.insert_file.empty_value")
            return None

        row = {
            "file_type": value.file_type,
            "disk": value.disk,
            "path": value.path,
            "original_filename": value.original_filename,
            "mime_type": value.mime_type,
            "extension": value.extension,
            "size_bytes": value.size_bytes,
            "media_link": value.media_link,
        }

        with self._db_pool.session() as session:
            try:
                result = session.execute(
                    insert(FileORM).returning(FileORM.id),
                    row,
                )
                file_id = result.scalar_one()
                session.commit()

                log.info(
                    "storage.attachment.insert_file.success",
                    file_id=str(file_id),
                )
                return file_id

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "storage.attachment.insert_file.error",
                    error=str(e),
                )
                raise e

    def insert_attachment(self, value: AttachmentORM) -> Optional[Any]:
        if not value:
            log.info("storage.attachment.insert_attachment.empty_value")
            return None

        row = {
            "file_id": value.file_id,
            "entity_type": value.entity_type,
            "entity_id": value.entity_id,
            "purpose": value.purpose,
        }

        with self._db_pool.session() as session:
            try:
                result = session.execute(
                    insert(AttachmentORM).returning(AttachmentORM.id),
                    row,
                )
                attachment_id = result.scalar_one()
                session.commit()

                log.info(
                    "storage.attachment.insert_attachment.success",
                    attachment_id=str(attachment_id),
                )
                return attachment_id

            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "storage.attachment.insert_attachment.error",
                    error=str(e),
                )
                raise e
