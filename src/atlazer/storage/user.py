import logging
import uuid

from uuid import UUID
from sqlalchemy import update, cast
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import UUID

from atlazer.storage.db import DatabasePool
from atlazer.models.user import ProfileUpdate, ProfileORM

logger = logging.getLogger(__name__)


class ProfileNotFoundError(Exception):
    """Raised when a profile can't be resolved to a single row via `id` unique
    constraint."""


class UserDepot:

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    def _values(self, payload: ProfileUpdate) -> dict:
        return {
            "interest_embedding": payload.interest_embedding,
            "next_processed_at": payload.next_processed_at
        }

    def update_profile(self, uuid_str: str, payload: ProfileUpdate) -> None:
        values = self._values(payload)
        if not values:
            return

        try:
            profile_uuid: UUID = uuid.UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {profile_uuid}")

        stmt = (
            update(ProfileORM) \
                .where(ProfileORM.id == profile_uuid) \
                .values(**values) \
                .returning(ProfileORM.id) \
                .execution_options(synchronize_session="fetch")
        )

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)
                updated_id = result.scalar()

                if updated_id is None:
                    session.rollback()
                    raise ProfileNotFoundError(
                        f"Could not update profile "
                        f"profile_id={profile_uuid!r} "
                        "(no matching row)"
                    )

                session.commit()
        except SQLAlchemyError:
            logger.exception("Failed to update profile id=%s", profile_uuid)
            raise
