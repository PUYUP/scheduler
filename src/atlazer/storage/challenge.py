import structlog
import uuid

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Any
from uuid import UUID

from atlazer.storage.db import DatabasePool
from atlazer.models.challenge import ChallengeORM, ChallengePaperORM
from atlazer.models.paper import PaperORM
from sqlalchemy import insert

log = structlog.get_logger(__name__)


class ChallengeDepot:

    def __init__(self, db_pool: DatabasePool):
        self._db_pool = db_pool

    def insert_challenge(
        self,
        user_id: str,
        target_date: date,
        papers: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> ChallengeORM:
        """
        Args:
            user_id: UUID string user pemilik challenge.
            target_date: tanggal target challenge.
            papers: hasil dari `MatcherDepot.match_papers_by_interest`, berbentuk
                {
                    "closest": [{"paper": PaperORM, "distance": float, "relevance_score": float}],
                    "farthest": [{"paper": PaperORM, "distance": float, "relevance_score": float}],
                }
                Boleh None / kosong jika tidak ada paper yang mau dilampirkan.
        """
        log.info("challenge.start_insert", user_id=user_id, papers=papers)

        try:
            user_uuid: UUID = uuid.UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {user_id}")

        with self._db_pool.session() as session:
            try:
                new_challenge = ChallengeORM(
                    user_id=user_uuid,
                    target_date=target_date,
                )
                session.add(new_challenge)
                session.flush()  # supaya new_challenge.id sudah terisi

                if papers:
                    challenge_paper_rows = self._build_challenge_paper_rows(
                        challenge_id=new_challenge.id,
                        papers=papers,
                    )

                    if challenge_paper_rows:
                        session.execute(insert(ChallengePaperORM), challenge_paper_rows)

                session.commit()
                session.refresh(new_challenge)
                log.info("challenge.finish_insert", user_id=user_id, papers=papers)
                return new_challenge
            except Exception as e:
                session.rollback()
                log.error(
                    "challenge.error_insert",
                    user_id=user_id,
                    papers=papers,
                    error=str(e),
                )
                raise e

    @staticmethod
    def _build_challenge_paper_rows(
        challenge_id: UUID,
        papers: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Ubah dict hasil matching yang sudah di-serialize
        (`{"closest": [{"id", "pdf_url", "title", "distance", "relevance_score"}, ...], "farthest": [...]}`)
        menjadi list of dict siap dipakai bulk insert ke tabel `challenge_papers`.
        """
        rows: List[Dict[str, Any]] = []

        for label in ("closest", "farthest"):
            matches = papers.get(label) or []
            for match in matches:
                paper_id = match.get("id")
                if paper_id is None:
                    log.warning(
                        "challenge.build_rows.missing_paper_id",
                        label=label,
                        match=match,
                    )
                    continue

                relevance_score = match.get("relevance_score")

                rows.append(
                    {
                        "challenge_id": challenge_id,
                        "paper_id": paper_id,
                        "relevance_score": _to_decimal_score(relevance_score),
                        "relevance_label": label,
                    }
                )

        return rows


def _to_decimal_score(score: Optional[float]) -> Optional[Decimal]:
    """
    Konversi float relevance_score ke Decimal(3, 2) agar sesuai kolom
    `Numeric(3, 2)` dan CHECK constraint (0 <= score <= 1) di DB.
    """
    if score is None:
        return None
    return Decimal(str(score)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)