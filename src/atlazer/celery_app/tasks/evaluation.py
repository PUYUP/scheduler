from __future__ import annotations
import structlog

from typing import Dict, Any
from atlazer.celery_app.main import app, db_pool
from atlazer.storage.paper import PaperDepot
from atlazer.storage.challenge import ChallengeDepot
from atlazer.utils.answer_scoring import getting_answer_chunks

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 9 — evaluation_answers with critical thinking, etc
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.evaluate_answers",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def evaluate_answers(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 of 9 — generate jsonl for batching
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.generate_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="challenge",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def generate_jsonl(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("evaluation.generate_jsonl", metadata=metadata)

    paper_id = metadata.get("paper_id")
    challenge_id = metadata.get("challenge_id")
    answer_id = metadata.get("answer_id")
    user_id = metadata.get("user_id")

    if not paper_id or not challenge_id or not answer_id or not user_id:
        raise ValueError("Missing required ids in metadata")

    # paper
    log.info("evaluation.generate_jsonl.paper", paper_id=paper_id)
    paper_depot = PaperDepot(db_pool)
    paper = paper_depot.get_paper_by_id(paper_id)
    if paper is None:
        raise ValueError(f"Paper with id {paper_id} not found")

    paper_chunks = paper_depot.get_chunks_by_paper_id(paper_id)
    if paper_chunks is None:
        raise ValueError(f"Paper with id {paper_id} not found")

    paper_contents = [
        f"{c.section}\n{c.content}"
        for c in paper_chunks if c.content is not None
    ]

    # answer
    log.info("evaluation.generate_jsonl.answer", answer_id=answer_id)
    challenge_depot = ChallengeDepot(db_pool)
    answer = challenge_depot.get_answer_by_id(answer_id)
    if answer is None:
        raise ValueError(f"Answer with id {answer_id} not found")

    log.info("evaluation.generate_jsonl.answer_chunks", answer_id=answer_id)
    answer_similarities = challenge_depot.get_answer_similarities_by_answer_id(answer_id)
    if answer_similarities is None:
        raise ValueError(f"Answer similarities for answer {answer_id} not found")

    answer_contents = [
        f"**Paper Chunk:** {c.paper_chunk_content}\n**Answer Chunk:** {c.answer_chunk_content}\n" +
        f"**Similarity Score:** {c.similarity_score}" if c.similarity_score is not None else ""
        for c in answer_similarities if c.answer_chunk_content is not None
    ]

    payload = {
        "key": f"evaluate/{user_id}/{challenge_id}/{answer_id}",
        "request": {
            "contents": [
                {
                    "text": f"""
                        **Paper Title:** {paper.title}\n\n
                        **Abstract:** {paper.abstract}\n\n
                        **Paper Content:** {"\n---\n".join(paper_contents)}\n\n
                        **Answer Content:** {answer.content}\n\n
                        **Answer Similarity With Paper Chunk:** {"\n---\n".join(answer_contents)}
                    """
                }
            ],
            "generation_config": {
                
            }
        }
    }

    log.info("evaluation.generate_jsonl.payload", payload=payload)

    return metadata