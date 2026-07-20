from __future__ import annotations
import structlog
import json

from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any
from atlazer.celery_app.main import app, db_pool
from atlazer.storage.paper import PaperDepot
from atlazer.storage.challenge import ChallengeDepot
from atlazer.models.evaluation import EvaluationORM
from atlazer.config.settings import settings
from atlazer.utils.gemini_batch import upload_chunk_file, scoring_chunk_file, get_batch_results
from atlazer.models.evaluation import CognitiveAssessment

log = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 9 — generate jsonl for batching
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.generate_jsonl",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="evaluation",
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
    language_code = metadata.get("language_code")

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

    key = f"evaluate/{user_id}/{challenge_id}/{answer_id}"
    payload = {
        "key": key,
        "request": {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""
                                You are a Neuro-Cognitive Evaluator and Radical Academic Mentor. Your task is not merely to grade answers, but to rigorously assess Higher-Order Thinking Skills (HOTS) and stretch the user's cognitive capacity to its absolute limit to stimulate neuroplasticity.
                                
                                **CRITICAL LANGUAGE RULE:**
                                1. You must generate all natural language text (such as explanations, feedback, questions, and list items) strictly in the language specified by the variable: {language_code}. Do not translate the JSON keys or structural metrics.
                                2. **DO NOT translate scientific, technical, or academic terminology.** Keep all domain-specific nomenclature (including SOLO Taxonomy levels) in its original, universally accepted form to preserve precise scientific meaning.
                                
                                Critically evaluate the user's answer based on these strict criteria:
                                1. Content Mastery and Accuracy (1-10): Assess the depth of understanding and factual accuracy regarding the paper's core scientific concepts and data.
                                2. Logical Flow of Arguments (1-10): Evaluate the structural coherence, logical progression, and analytical rigor of the arguments presented.
                                3. Critical Thinking and Analysis (1-10): Measure the ability to deconstruct assumptions, evaluate methodologies, and grasp the systemic implications of the paper (beyond mere summarization).
                                4. Academic Language and Writing Mechanics (1-10): Grade the precision of terminology, scholarly tone, clarity, and articulation of complex ideas.
                                5. Cognitive Synthesis & Connectivity (1-10): Assess the ability to connect disparate, complex concepts within the paper to form sharp, novel insights.
                                6. Structural Memory & Retrieval (1-10): Evaluate whether the user successfully integrated crucial structural details from the paper, or if the retrieval was only surface-level.
                                7. Cognitive Stretch / Edge (1-10): Determine if this answer demonstrates maximum intellectual effort, or if the user played it safe. How hard was their brain forced to think?
                                
                                **SOLO TAXONOMY EVALUATION (0-10 SCALE):**
                                Evaluate the user's mastery of each level in the Structure of the Observed Learning Outcome (SOLO) Taxonomy on a scale of 0 to 10. Generate a cognitive footprint by scoring how strongly the answer fulfills the criteria of each stage:
                                - "Prestructural" (0-10): Score high ONLY if the answer completely misses the point, uses irrelevant information, or shows deep misunderstanding (A highly competent answer should score 0 here).
                                - "Unistructural" (0-10): Score the ability to correctly identify and focus on at least one relevant, standalone aspect.
                                - "Multistructural" (0-10): Score the ability to identify and list multiple relevant aspects accurately, even if treated independently.
                                - "Relational" (0-10): Score the depth and ability to integrate various aspects into a cohesive, logically interconnected structure.
                                - "Extended Abstract" (0-10): Score the ability to generalize the integrated structure to propose hypotheses, new domains, or novel theoretical implications beyond the provided text.

                                Analyze the following dataset rigidly:

                                **Paper Title:** 
                                {paper.title}

                                **Abstract:** 
                                {paper.abstract}

                                **Paper Content (Chunks):** 
                                {"\n---\n".join(paper_contents)}

                                **User Answer Content:** 
                                {answer.content}

                                **Answer Similarity With Paper Chunk:** 
                                {"\n---\n".join(answer_contents)}
                                
                                Return your evaluation strictly as a raw, pure JSON object matching this schema exactly. Ensure the text values within the JSON adhere strictly to the requested language ({language_code}) while keeping scientific terms untranslated:
                                {{
                                    "content_mastery_score": 0,
                                    "logical_flow_score": 0,
                                    "critical_thinking_score": 0,
                                    "academic_language_score": 0,
                                    "cognitive_synthesis_score": 0,
                                    "memory_retention_score": 0,
                                    "cognitive_stretch_score": 0,
                                    "solo": {{
                                        "prestructural": 0,
                                        "unistructural": 0,
                                        "multistructural": 0,
                                        "relational": 0,
                                        "extended_abstract": 0
                                    }},
                                    "blind_spots": ["List specific logical flaws, biases, misinterpretations, or overlooked nuances by the user"],
                                    "neuro_growth_feedback": "Provide an intellectually provocative, brutally honest, yet constructive critique. Pinpoint exactly where their reasoning softened and how they should have challenged the text.",
                                    "brain_hack_question": "Formulate one highly sophisticated, complex question based on this paper's deep mechanics to force the user to build new neural connections in their next session.",
                                    "international_metrics_evaluated": {{
                                        "issue_clarification_depth": null,
                                        "evidence_interrogativity": null,
                                        "assumption_and_bias_detection": null,
                                        "divergent_synthesis": null,
                                        "systemic_implication": null
                                    }}
                                }}
                            """
                        }
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0.15, 
                "top_p": 0.85,
                "max_output_tokens": 1500,
                "response_mime_type": "application/json",
                "response_schema": _resolve_pydantic_schema(CognitiveAssessment),
            }
        }
    }

    target_dir = Path(settings.gemini_batch_dir)
    target_file = target_dir / f"{key}.jsonl"

    # create full path of target file
    target_file.parent.mkdir(exist_ok=True, parents=True)

    log.info("evaluation.generate_jsonl.payload", target_dir=target_dir)
    with open(target_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # upload file
    file_name = upload_chunk_file(str(target_file), display_name=key)
    if file_name is None:
        raise ValueError(f"Failed to upload file {target_file}")

    log.info("evaluation.generate_jsonl.scoring_results", file_name=file_name)
    metadata["display_name"] = key
    metadata["file_name"] = file_name
    metadata["target_dir"] = str(target_dir)
    metadata["target_file"] = str(target_file)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 of 9 — scoring answer with critical thinking, etc
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.scoring_answer",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="evaluation",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def scoring_answer(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("evaluation.scoring_answer", metadata=metadata)

    target_file = metadata.get("target_file")
    if target_file is None:
        raise ValueError("Failed to get target file from metadata")

    file_name = metadata.get("file_name")
    if file_name is None:
        raise ValueError("Failed to get file name from metadata")

    # process to gemini AI
    user_metadata = {
        "user_id": metadata.get("user_id"),
        "answer_id": metadata.get("answer_id"),
        "challenge_id": metadata.get("challenge_id"),
        "challenge_paper_id": metadata.get("challenge_paper_id"),
        "paper_id": metadata.get("paper_id"),
        "action": "answer_score_generation",
    }

    job_name = scoring_chunk_file(
        file_name,
        model="gemini-3.1-flash-lite",
        user_metadata=user_metadata
    )

    if job_name is None:
        raise ValueError(f"Failed to create job for file {target_file}")

    log.info("evaluation.scoring_answer.scoring_ran", job_name=job_name)

    metadata["job_name"] = job_name
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 9 of 9 — save evaluatuion to database
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlazer.celery_app.tasks.evaluation.save_evaluation",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="evaluation",
    time_limit=1800,
    soft_time_limit=1700,
    ignore_result=False,
)
def save_evaluation(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    log.info("evaluation.save_evaluation", metadata=metadata)

    job_id = metadata.get("job_id")
    if job_id is None:
        raise ValueError("Failed to get job id from metadata")

    try:
        results = get_batch_results(job_id)
        if results is None:
            raise ValueError("Failed to get results from batch")
    except Exception as e:
        log.error("evaluation.save_evaluation.error", error=str(e))
        raise ValueError(str(e))

    log.info("evaluation.save_evaluation.batch_results", results=results)

    payload: EvaluationORM = EvaluationORM(
        user_id=metadata.get("user_id"),
        challenge_id=metadata.get("challenge_id"),
        challenge_paper_id=metadata.get("challenge_paper_id"),
        answer_id=metadata.get("answer_id"),
        results=results,
    )

    try:
        depot = ChallengeDepot(db_pool)
        depot.insert_evaluation(payload)
    except Exception as e:
        log.error("evaluation.save_evaluation.error", error=str(e))
        raise ValueError(str(e))

    log.info("evaluation.save_evaluation.success", metadata=metadata)
    return metadata


def _resolve_pydantic_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a Gemini-compatible schema dict
    (inlines $defs/$ref, karena Gemini response_schema tidak support $ref).
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                resolved = _resolve(defs[ref_key])
                # gabungkan field lain (misal 'description' di level pemanggil) jika ada
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**resolved, **extra}
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)
