from __future__ import annotations
import structlog

from typing import Dict, Any
from atlazer.celery_app.main import app, db_pool
from atlazer.storage.paper import PaperDepot

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

    paper_depot = PaperDepot(db_pool)
    paper_chunks = paper_depot.get_chunks_by_paper_id(paper_id)

    payload = {
        "key": f"evaluate/{user_id}/{challenge_id}/{answer_id}",
        "request": {
            "contents": [
                {
                    "text": "text is here"
                }
            ],
            "generation_config": {
                
            }
        }
    }
    
    return metadata
