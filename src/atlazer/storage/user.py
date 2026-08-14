import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, cast
from collections import defaultdict

from uuid import UUID
from sqlalchemy import select, update, or_, CursorResult, bindparam, Table
from sqlalchemy.exc import SQLAlchemyError

from atlazer.storage.db import DatabasePool
from atlazer.models.user import ProfileUpdate, ProfileORM, SubscriptionORM

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
            "next_processing_at": payload.next_processing_at
        }

    def get_profiles_for_paper_matching(self) -> List[Dict[str, Any]]:
        """Get profiles that are ready for paper matching."""
        current_time = datetime.now(timezone.utc).isoformat()

        # Gunakan .is_(None) yang merupakan standar SQLAlchemy untuk perbandingan NULL
        stmt = (
            select(ProfileORM, SubscriptionORM.attributes)
            .outerjoin(SubscriptionORM, SubscriptionORM.user_id == ProfileORM.user_id)
            .where(
                or_(
                    ProfileORM.next_processing_at.is_(None),
                    ProfileORM.next_processing_at < current_time,
                )
            )
            .limit(10)
        )

        with self._pool.session() as session:
            try:
                rows = session.execute(stmt).all()

                return [
                    {
                        "id": p.id,
                        "user_id": p.user_id,
                        "interest": p.interest,
                        "interest_embedding": p.interest_embedding,
                        "language_code": p.language_code,
                        "next_processing_at": p.next_processing_at.isoformat() if p.next_processing_at else None,
                        "subscription_attributes": attributes if attributes is not None else {},
                    }
                    for p, attributes in rows
                ]
            except Exception as e:
                # Menggunakan logger.exception agar merekam stack-trace penuh
                logger.exception("matcher.paper_for_user.failed: %s", str(e))
                raise

    def get_profile(self, uuid_str: str) -> ProfileORM:
        """Mengambil data profile berdasarkan UUID string."""
        try:
            profile_uuid: UUID = uuid.UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {uuid_str}")

        stmt = select(ProfileORM).where(ProfileORM.id == profile_uuid)

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)
                profile = result.scalar_one_or_none()

                if profile is None:
                    raise ProfileNotFoundError(
                        f"Could not find profile "
                        f"profile_id={profile_uuid!r}"
                    )

                return profile
        except SQLAlchemyError:
            logger.exception("Failed to fetch profile id=%s", profile_uuid)
            raise
    
    def get_profile_by_user_id(self, user_id: str) -> ProfileORM:
        """Mengambil data profile berdasarkan User ID string."""
        try:
            user_uuid: UUID = uuid.UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {user_id}")

        stmt = select(ProfileORM).where(ProfileORM.user_id == user_uuid)

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)
                profile = result.scalar_one_or_none()

                if profile is None:
                    raise ProfileNotFoundError(
                        f"Could not find profile "
                        f"user_id={user_uuid!r}"
                    )

                return profile
        except SQLAlchemyError:
            logger.exception("Failed to fetch profile from user_id=%s", user_uuid)
            raise

    def update_profile(self, uuid_str: str, payload: ProfileUpdate) -> None:
        values = self._values(payload)
        if not values:
            return

        try:
            profile_uuid: UUID = uuid.UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {uuid_str}")

        stmt = (
            update(ProfileORM)
                .where(ProfileORM.id == profile_uuid)
                .values(**values)
                .returning(ProfileORM.id)
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

    def bulk_update_profiles(self, updates: List[Tuple[str, ProfileUpdate]]) -> int:
        """
        Bulk update profile dengan payload berbeda per profile,
        dikelompokkan berdasarkan field yang sama untuk efisiensi.
        """
        if not updates:
            return 0

        grouped: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)

        for uuid_str, payload in updates:
            try:
                profile_id = uuid.UUID(uuid_str)
            except ValueError:
                raise ValueError(f"Invalid UUID string format: {uuid_str}")

            raw_values = self._values(payload)
            values = {k: v for k, v in raw_values.items() if v is not None}

            if not values:
                continue

            key = frozenset(values.keys())
            values["_id"] = profile_id  # nama beda dari kolom asli, dipakai di WHERE
            grouped[key].append(values)

        if not grouped:
            return 0

        total_updated = 0
        try:
            with self._pool.session() as session:
                profile_table = cast(Table, ProfileORM.__table__)

                for field_keys, rows in grouped.items():
                    stmt = (
                        update(profile_table)
                        .where(profile_table.c.id == bindparam("_id"))
                        .values({field: bindparam(field) for field in field_keys})
                    )
                    result = session.execute(stmt, rows)
                    if isinstance(result, CursorResult):
                        total_updated += result.rowcount

                session.commit()

            logger.info("Bulk updated %d profiles (grouped by field set)", total_updated)
            return total_updated

        except SQLAlchemyError:
            logger.exception("Failed to bulk update profiles with per-user payloads")
            raise
