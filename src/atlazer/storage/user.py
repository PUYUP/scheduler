import logging

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

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
        }

    def update_profile(self, uuid: str, payload: ProfileUpdate) -> None:
        values = self._values(payload)
        if not values:
            return

        stmt = (
            update(ProfileORM) \
                .where(ProfileORM.id == uuid) \
                .values(**values) \
                .execution_options(synchronize_session="fetch")
        )

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)

                if result.rowcount == 0:
                    session.rollback()
                    raise ProfileNotFoundError(
                        f"Could not update profile "
                        f"profile_id={uuid!r} "
                        "(no matching row)"
                    )

                session.commit()
        except SQLAlchemyError:
            logger.exception("Failed to update profile id=%s", uuid)
            raise
