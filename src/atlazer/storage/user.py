import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

from uuid import UUID
from sqlalchemy import select, update, or_, CursorResult
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
            "next_processed_at": payload.next_processed_at
        }

    def get_profiles_for_paper_matching(self) -> List[Dict[str, Any]]:
        """Get profiles that are ready for paper matching."""
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Gunakan .is_(None) yang merupakan standar SQLAlchemy untuk perbandingan NULL
        stmt = select(ProfileORM).where(
            or_(
                ProfileORM.next_processed_at.is_(None),
                ProfileORM.next_processed_at < current_time
            )
        ).limit(10)

        with self._pool.session() as session:
            try:
                profiles = session.execute(stmt).scalars().all()

                return [
                    {
                        "id": p.id,
                        "user_id": p.user_id,
                        "interest": p.interest,
                        "interest_embedding": p.interest_embedding,
                        "language_code": p.language_code,
                        "next_processed_at": p.next_processed_at.isoformat() if p.next_processed_at else None,
                    }
                    for p in profiles
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

    def bulk_update_profiles(self, uuid_strs: List[str], payload: ProfileUpdate) -> int:
        """
        Melakukan bulk update pada beberapa profile sekaligus berdasarkan list ID.
        Mengembalikan jumlah baris (rowcount) yang berhasil di-update.
        """
        if not uuid_strs:
            return 0

        # Ambil dictionary dari payload
        raw_values = self._values(payload)
        
        # FILTERING: Hanya simpan field yang TIDAK None
        values = {k: v for k, v in raw_values.items() if v is not None}

        # Jika setelah difilter ternyata kosong (tidak ada yang perlu diupdate), hentikan
        if not values:
            return 0

        profile_uuids: List[UUID] = []
        for uuid_str in uuid_strs:
            try:
                profile_uuids.append(uuid.UUID(uuid_str))
            except ValueError:
                raise ValueError(f"Invalid UUID string format in list: {uuid_str}")

        stmt = (
            update(ProfileORM)
            .where(ProfileORM.id.in_(profile_uuids))
            .values(**values) # Sekarang hanya berisi field yang benar-benar ada nilainya
            .execution_options(synchronize_session="fetch")
        )

        try:
            with self._pool.session() as session:
                result = session.execute(stmt)
                session.commit()
                
                if isinstance(result, CursorResult):
                    updated_count = result.rowcount
                else:
                    updated_count = 0

                logger.info("Bulk updated %d profiles", updated_count)
                
                return updated_count
        except SQLAlchemyError:
            logger.exception("Failed to bulk update profiles for %d IDs", len(uuid_strs))
            raise