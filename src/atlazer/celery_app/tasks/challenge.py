from __future__ import annotations

import uuid
import structlog
import numpy as np
from typing import Dict, Any, List
from celery import group, signature
from sklearn.metrics.pairwise import cosine_similarity

from atlazer.celery_app.main import app, db_pool
from atlazer.utils.stanza_chunker import chunk_answer as stanza_chunk_answer
from atlazer.config.settings import settings
from atlazer.utils.embedder import chunks_to_vector
from atlazer.storage.challenge import ChallengeDepot
from atlazer.models.challenge import (
    ChunkAnswerMetadata,
    AnswerChunkORM,
    AnswerSimilarityORM
)
from atlazer.utils.answer_scoring import (
    getting_answer_chunks,
    getting_paper_chunks,
)

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 7 — chunk_answer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.chunk_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def chunk_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    validated = ChunkAnswerMetadata.model_validate(metadata)
    content = validated.content
    language_code = validated.language_code

    log.info("challenge.chunk_answer.start", metadata=validated.model_dump())

    chunks = stanza_chunk_answer(
        text=content,
        lang=language_code,
        semantic=True,
        download_models=False,
        embed_model_name=settings.local_embedding_model,
        min_words=1,
        max_words=35,
    )

    validated.chunks = [{"text": chunk} for chunk in chunks]

    log.info(
        "challenge.chunk_answer.done",
        chunk_count=len(validated.chunks),
        metadata=validated.model_dump()
    )

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 7 — embed_answer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.embed_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def embed_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates embedding vectors for every chunk in `metadata["chunks"]`.

    Chunks are batched to stay within API token limits.
    Each chunk is enriched with an `embedding` key (list[float]).

    Returns the full metadata dict with embedded chunks.
    """
    chunks = metadata.get("chunks", [])
    embedded_chunks: List[Dict[str, Any]] = []

    if not chunks:
        log.warning("challenge.embed_answer.no_chunks", metadata=metadata)
        return metadata

    log.info(
        "challenge.embed_answer.start",
        chunks=len(chunks),
        provider=settings.embedding_provider,
        metadata=metadata,
    )

    try:
        embedded_chunks = chunks_to_vector(chunks)
    except Exception as exc:
        log.error(
            "challenge.embed_answer.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "challenge.embed_answer.done",
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
        metadata=metadata,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 7 — save embedding answer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.save_embedding_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_embedding_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("challenge.save_embedding_answer.start")

    chunks = metadata.get('chunks')
    user_id = metadata.get('user_id')
    challenge_id = metadata.get('challenge_id')
    answer_id = metadata.get('answer_id')

    if not chunks:
        log.warning("challenge.save_embedding_answer.no_chunks", metadata=metadata)
        raise ValueError("No chunks to save")

    if not user_id or not challenge_id:
        log.warning("challenge.save_embedding_answer.missing_user_id_or_challenge_id", metadata=metadata)
        raise ValueError("Missing user_id or challenge_id")

    if not answer_id:
        log.warning("challenge.save_embedding_answer.missing_answer_id", metadata=metadata)
        raise ValueError("Missing answer_id")

    depot = ChallengeDepot(db_pool)

    try:
        user_uuid = uuid.UUID(str(user_id))
        challenge_uuid = uuid.UUID(str(challenge_id))
    except ValueError as exc:
        log.error("challenge.save_embedding_answer.invalid_uuid", metadata=metadata, error=str(exc))
        raise ValueError("Invalid UUID string format for user_id or challenge_id")

    log.info("challenge.save_embedding_answer.mapping_payloads")
    payloads: List[AnswerChunkORM] = []

    for chunk in chunks:
        payloads.append(
            AnswerChunkORM(
                user_id=user_uuid,
                challenge_id=challenge_uuid,
                answer_id=answer_id,
                content=chunk.get("text"),
                embedding=chunk.get("embedding"),
                embedding_model=chunk.get("embedding_model"),
                embedding_adapter=chunk.get("embedding_adapter"),
                embedding_normalized=chunk.get("embedding_normalized", True),
                token_count=chunk.get("token_count"),
                word_count=chunk.get("word_count"),
            )
        )

    try:
        depot.bulk_insert_answer_chunks(payloads)
    except Exception as exc:
        log.error(
            "challenge.save_embedding_answer.failed",
            metadata=metadata,
            error=str(exc),
            attempt=self.request.retries,
        )
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info("challenge.save_embedding_answer.done")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 7 — answer scoring
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.answer_scoring",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def answer_scoring(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("challenge.answer_scoring.start")
    challenge_id = metadata.get("challenge_id")
    answer_id = metadata.get("answer_id")

    if not challenge_id:
        log.warning("challenge.answer_scoring.missing_challenge_id", metadata=metadata)
        raise ValueError("Missing challenge_id")

    if not answer_id:
        log.warning("challenge.answer_scoring.missing_answer_id", metadata=metadata)
        raise ValueError("Missing answer_id")

    log.info(
        'challenge.answer_scoring.get_answer_vectors',
        answer_id=answer_id
    )

    answer_vectors = getting_answer_chunks(answer_id)
    if not answer_vectors:
        log.warning("challenge.answer_scoring.no_answer_vectors", metadata=metadata)
        raise ValueError("No answer vectors found")

    # Get challenge papers
    depot = ChallengeDepot(db_pool)
    challenge_papers = depot.get_challenge_papers_by_challenge_id(challenge_id)
    if not challenge_papers:
        log.warning("challenge.answer_scoring.no_challenge_papers", metadata=metadata)
        raise ValueError("No challenge papers found")

    job = group(
        answer_paper_scoring.s({
            "paper_id": str(cp.paper_id),
            "challenge_paper_id": str(cp.id),
            "answer_vectors": answer_vectors,
            **metadata
        }).set(queue="challenge") for cp in challenge_papers
    )
    job.apply_async()
    
    metadata["answer_vectors"] = answer_vectors
    metadata["challenge_papers"] = [{"id": str(x.id), "paper_id": str(x.paper_id)} for x in challenge_papers]
    
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 7 — answer + paper scoring
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.answer_paper_scoring",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def answer_paper_scoring(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("challenge.answer_paper_scoring.start")

    paper_id = metadata.get("paper_id")
    answer_vectors = metadata.get("answer_vectors")

    if not paper_id:
        log.warning("challenge.answer_paper_scoring.missing_paper_id", metadata=metadata)
        raise ValueError("Missing paper_id")

    if not answer_vectors:
        log.warning("challenge.answer_paper_scoring.missing_answer_vectors", metadata=metadata)
        raise ValueError("Missing answer vectors")

    log.info(
        'challenge.answer_paper_scoring.get_paper_vectors',
        paper_id=paper_id
    )

    paper_vectors = getting_paper_chunks(paper_id)
    paper_embeddings = [x["embedding"] for x in paper_vectors]
    answer_embeddings = [x["embedding"] for x in answer_vectors]

    log.info(
        'challenge.answer_paper_scoring.calculating_similarity',
        answer_embeddings=len(answer_embeddings),
        paper_embeddings=len(paper_embeddings)
    )

    similarity_matrix = cosine_similarity(answer_embeddings, paper_embeddings)
    data_to_insert = []

    for i, chunk in enumerate(answer_vectors):
        scores_for_c = similarity_matrix[i]
        best_match_index = np.argmax(scores_for_c)
        highest_score = scores_for_c[best_match_index]
        paper = paper_vectors[best_match_index]

        data_to_insert.append({
            "answer_chunk_id": chunk.get("id"),
            "answer_chunk_content": chunk.get("content"),
            "document_chunk_id": paper.get("id"),
            "paper_chunk_content": paper.get("content"),
            "similarity_score": float(highest_score),
        })

    if data_to_insert:
        save_answer_similarity.apply_async(
            kwargs={
                "metadata": {
                    "user_id": metadata.get("user_id"),
                    "challenge_id": metadata.get("challenge_id"),
                    "challenge_paper_id": metadata.get("challenge_paper_id"),
                    "answer_id": metadata.get("answer_id"),
                    "paper_id": paper_id,
                    "data_to_insert": data_to_insert
                }
            },
            queue="challenge"
        )

    log.info(
        'challenge.answer_paper_scoring.similarity_matrix_calculated',
        similarity_matrix=similarity_matrix.shape,
        data_to_insert=len(data_to_insert)
    )

    metadata["data_to_insert"] = data_to_insert
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 7 — save answer similarity
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.challenge.save_answer_similarity",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_answer_similarity(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("challenge.save_answer_similarity.start")
    data_to_insert = metadata.get("data_to_insert", [])
    payloads: List[AnswerSimilarityORM] = []
    depot = ChallengeDepot(db_pool)

    for data in data_to_insert:
        payload = AnswerSimilarityORM(
            answer_id=metadata.get("answer_id"),
            challenge_id=metadata.get("challenge_id"),
            challenge_paper_id=metadata.get("challenge_paper_id"),
            answer_chunk_id=data.get("answer_chunk_id"),
            document_chunk_id=data.get("document_chunk_id"),
            paper_id=metadata.get("paper_id"),
            user_id=metadata.get("user_id"),
            answer_chunk_content=data.get("answer_chunk_content"),
            paper_chunk_content=data.get("paper_chunk_content"),
            similarity_score=data.get("similarity_score"),
        )
        payloads.append(payload)

    if payloads:
        depot.bulk_inser_answer_similarities(payloads)

    log.info(
        'challenge.save_answer_similarity.success',
        payloads_count=len(payloads)
    )

    # Chain: save_answer_similarity → evaluate_answers (evaluation queue)
    (
        signature(
            "atlazer.celery_app.tasks.evaluation.generate_jsonl",
            args=(metadata,),
            queue="evaluation",
            immutable=False,
        )
        |
        signature(
            "atlazer.celery_app.tasks.evaluation.scoring_answer",
            args=(metadata,),
            queue="evaluation",
            immutable=False,
        )
    ).apply_async()

    return metadata