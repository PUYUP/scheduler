import structlog
import uuid

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Any
from uuid import UUID

from atlazer.storage.db import DatabasePool
from atlazer.models.challenge import (
    ChallengeORM,
    ChallengePaperORM,
    PaperSummaryORM,
    AnswerChunkORM,
    AnswerSimilarityORM
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert, select, tuple_

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

    def update_challenge_paper(
        self,
        challenge_paper_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[ChallengePaperORM]:
        """
        Memperbarui data ChallengePaperORM berdasarkan ID.
        Mendukung partial update (PATCH). Hanya field yang ada di `update_data`
        yang akan diubah.
        
        Args:
            challenge_paper_id: UUID string dari challenge_paper yang ingin diubah.
            update_data: Dictionary berisi field yang akan di-update, 
                         contoh: {"relevance_label": "closest", "relevance_score": 0.85}
        """
        log.info("challenge_paper.start_update", id=challenge_paper_id, payload=update_data)

        try:
            cp_uuid: UUID = uuid.UUID(challenge_paper_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {challenge_paper_id}")

        with self._db_pool.session() as session:
            try:
                # 1. Cari record yang ada
                record = session.query(ChallengePaperORM).filter(
                    ChallengePaperORM.id == cp_uuid
                ).first()

                if not record:
                    log.warning("challenge_paper.not_found", id=challenge_paper_id)
                    return None

                # 2. Lakukan iterasi update_data dan lakukan patch (partial update)
                for key, value in update_data.items():
                    # Format nilai khusus agar konsisten dengan DB (jika field ada di payload)
                    if key == "relevance_score":
                        value = _to_decimal_score(value)

                    # Update atribut secara dinamis jika properti tersebut ada di model
                    if hasattr(record, key):
                        setattr(record, key, value)
                    else:
                        log.debug("challenge_paper.ignore_unknown_field", field=key)

                # 3. Simpan perubahan
                session.commit()
                session.refresh(record)
                
                log.info("challenge_paper.finish_update", id=challenge_paper_id)
                return record

            except Exception as e:
                session.rollback()
                log.error(
                    "challenge_paper.error_update",
                    id=challenge_paper_id,
                    payload=update_data,
                    error=str(e),
                )
                raise e

    def insert_challenge_paper_summary(
        self, 
        data: Dict[str, Any],
    ) -> Optional[PaperSummaryORM]:
        """
        Simpan PaperSummaryORM ke database. Datanya berasal dari pemprosesan summary
        di ChallengePaper.

        Args:
            data: Dictionary berisi data PaperSummaryORM.
        
        Returns:
            PaperSummaryORM yang sudah disimpan.
        """
        with self._db_pool.session() as session:
            try:
                record = PaperSummaryORM(**data)
                session.add(record)
                session.commit()
                session.refresh(record)

                log.info("challenge_paper_summary.finish_insert", id=record.id)
                return record

            except Exception as e:
                session.rollback()
                log.error(
                    "challenge_paper_summary.error_insert",
                    payload=data,
                    error=str(e),
                )
                raise e    
    
    def update_challenge_paper_summary(
        self,
        challenge_paper_summary_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[PaperSummaryORM]:
        """
        Memperbarui data ChallengePaperORM berdasarkan ID.
        Mendukung partial update (PATCH). Hanya field yang ada di `update_data`
        yang akan diubah.
        
        Args:
            challenge_paper_summary_id: UUID string dari challenge_paper_summary yang ingin diubah.
            update_data: Dictionary berisi field yang akan di-update, 
                         contoh: {"relevance_label": "closest", "relevance_score": 0.85}
        """
        log.info("challenge_paper_summary.start_update", id=challenge_paper_summary_id, payload=update_data)

        try:
            cps_uuid: UUID = uuid.UUID(challenge_paper_summary_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {challenge_paper_summary_id}")

        with self._db_pool.session() as session:
            try:
                # 1. Cari record yang ada
                record = session.query(PaperSummaryORM).filter(
                    PaperSummaryORM.id == cps_uuid
                ).first()

                if not record:
                    log.warning("challenge_paper_summary.not_found", id=challenge_paper_summary_id)
                    return None

                # 2. Lakukan iterasi update_data dan lakukan patch (partial update)
                for key, value in update_data.items():
                    # Update atribut secara dinamis jika properti tersebut ada di model
                    if hasattr(record, key):
                        setattr(record, key, value)
                    else:
                        log.debug("challenge_paper_summary.ignore_unknown_field", field=key)

                # 3. Simpan perubahan
                session.commit()
                session.refresh(record)
                
                log.info("challenge_paper_summary.finish_update", id=challenge_paper_summary_id)
                return record

            except Exception as e:
                session.rollback()
                log.error(
                    "challenge_paper_summary.error_update",
                    id=challenge_paper_summary_id,
                    payload=update_data,
                    error=str(e),
                )
                raise e

    def save_embedding_answer(
        self, 
        data: AnswerChunkORM
    ) -> Dict[str, Any]:
        """
        Simpan embedding answer ke database.
        """
        with self._db_pool.session() as session:
            try:
                session.add(data)
                session.commit()
                session.refresh(data)

                log.info("answer_chunk.finish_insert", id=str(data.id))
                return {
                    "id": str(data.id),
                    "user_id": str(data.user_id),
                    "challenge_id": str(data.challenge_id),
                }

            except Exception as e:
                session.rollback()
                log.error(
                    "answer_chunk.error_insert",
                    error=str(e),
                )
                raise e

    def bulk_insert_answer_chunks(self, values: List[AnswerChunkORM]) -> None:
        if not values:
            log.info("answer_chunk.empty_list")
            return

        # 1. Ambil kombinasi unik dari (user_id, challenge_id)
        # Menggunakan Set {} agar jika ada kombinasi yang berulang tidak duplikat
        user_challenge_pairs = list({(chunk.user_id, chunk.challenge_id, chunk.answer_id) for chunk in values})

        # 2. Siapkan data baru untuk di-insert
        rows = [
            {
                "user_id": chunk.user_id,
                "challenge_id": chunk.challenge_id,
                "answer_id": chunk.answer_id,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "embedding_model": chunk.embedding_model,
                "embedding_adapter": chunk.embedding_adapter,
                "embedding_normalized": chunk.embedding_normalized,
                "token_count": chunk.token_count,
                "word_count": chunk.word_count,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                # 3. HAPUS chunk lama berdasarkan kombinasi (user_id DAN challenge_id)
                session.query(AnswerChunkORM).filter(
                    tuple_(
                        AnswerChunkORM.user_id, 
                        AnswerChunkORM.challenge_id,
                        AnswerChunkORM.answer_id
                    ).in_(user_challenge_pairs)
                ).delete(synchronize_session=False)

                # 4. INSERT data baru secara massal
                session.execute(insert(AnswerChunkORM), rows)
                
                session.commit()
                
                log.info(
                    "challenge_answer.finish_reindex",
                    user_challenge_pairs=str(user_challenge_pairs),
                    count=len(values)
                )
                
            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "challenge_answer.error_reindex",
                    error=str(e),
                )
                raise e

    def bulk_inser_answer_similarities(self, values: List[AnswerSimilarityORM]) -> None:
        if not values:
            log.info("answer_similarity.empty_list")
            return

        # 1. Ambil kombinasi unik dari (user_id, challenge_id)
        # Menggunakan Set {} agar jika ada kombinasi yang berulang tidak duplikat
        user_challenge_pairs = list({(chunk.user_id, chunk.challenge_id, chunk.answer_id, chunk.paper_id) for chunk in values})

        # 2. Siapkan data baru untuk di-insert
        rows = [
            {
                "user_id": chunk.user_id,
                "challenge_id": chunk.challenge_id,
                "challenge_paper_id": chunk.challenge_paper_id,
                "answer_id": chunk.answer_id,
                "answer_chunk_id": chunk.answer_chunk_id,
                "answer_chunk_content": chunk.answer_chunk_content,
                "paper_id": chunk.paper_id,
                "document_chunk_id": chunk.document_chunk_id,
                "paper_chunk_content": chunk.paper_chunk_content,
                "similarity_score": chunk.similarity_score,
            }
            for chunk in values
        ]

        with self._db_pool.session() as session:
            try:
                # 3. HAPUS chunk lama berdasarkan kombinasi (user_id DAN challenge_id)
                session.query(AnswerSimilarityORM).filter(
                    tuple_(
                        AnswerSimilarityORM.user_id, 
                        AnswerSimilarityORM.challenge_id,
                        AnswerSimilarityORM.answer_id,
                        AnswerSimilarityORM.paper_id
                    ).in_(user_challenge_pairs)
                ).delete(synchronize_session=False)

                # 4. INSERT data baru secara massal
                session.execute(insert(AnswerSimilarityORM), rows)
                
                session.commit()
                
                log.info(
                    "challenge_answer_similarity.finish_reindex",
                    user_challenge_pairs=str(user_challenge_pairs),
                    count=len(values)
                )
                
            except SQLAlchemyError as e:
                session.rollback()
                log.error(
                    "challenge_answer_similarity.error_reindex",
                    error=str(e),
                )
                raise e

    def get_chunks_by_answer_id(self, answer_id: str) -> List[AnswerChunkORM] | None:
        """Return chunks of an answer."""
        try:
            answer_uuid: UUID = uuid.UUID(answer_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {answer_id}")

        stmt = (
            select(AnswerChunkORM)
            .where(AnswerChunkORM.answer_id == answer_uuid)
        )

        with self._db_pool.session() as session:
            rows = session.execute(stmt).scalars().all()
            return list(rows)

    def get_challenge_papers_by_challenge_id(self, challenge_id: str) -> List[ChallengePaperORM]:
        try:
            challenge_uuid: UUID = uuid.UUID(challenge_id)
        except ValueError:
            raise ValueError(f"Invalid UUID string format: {challenge_id}")

        stmt = (
            select(ChallengePaperORM)
            .where(ChallengePaperORM.challenge_id == challenge_uuid)
        )

        with self._db_pool.session() as session:
            rows = session.execute(stmt).scalars().all()
            return list(rows)

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