from sqlalchemy import TIMESTAMP
from sqlalchemy import Index
from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Mapped
from pydantic import BaseModel, Field
from typing import List
from atlazer.models.base import Base
from typing import Optional, List, Dict, Any
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from uuid import UUID
from sqlalchemy.sql import func


class InternationalMetricsEvaluated(BaseModel):
    issue_clarification_depth: str | None = Field(
        ...,
        description=(
            "Evaluate the sharpness in unraveling complex issues within the paper. "
            "Set to null if the user did not attempt this task at all, "
            "did not engage with the paper's core issues, or gave no substantive response."
        )
    )
    evidence_interrogativity: str | None = Field(
        ...,
        description=(
            "Evaluate whether the user aggressively questioned the data/methodologies "
            "rather than passively summarizing. Set to null if the user did not "
            "attempt this task at all, did not engage with the paper's core issues, "
            "or gave no substantive response."
        )
    )
    assumption_and_bias_detection: str | None = Field(
        ...,
        description=(
            "Evaluate the ability to detect the author's underlying assumptions and "
            "the user's own cognitive biases. Set to null if the user did not attempt "
            "this task at all, did not engage with the paper's core issues, or gave "
            "no substantive response."
        )
    )
    divergent_synthesis: str | None = Field(
        ...,
        description=(
            "Evaluate the capacity to bridge fragmented pieces of information into a "
            "novel, overarching perspective. Set to null if the user did not attempt "
            "this task at all, did not engage with the paper's core issues, or gave "
            "no substantive response."
        )
    )
    systemic_implication: str | None = Field(
        ...,
        description=(
            "Evaluate the foresight in predicting long-term, real-world impacts or "
            "consequences of the paper's findings. Set to null if the user did not "
            "attempt this task at all, did not engage with the paper's core issues, "
            "or gave no substantive response."
        )
    )


class SOLOTaxonomy(BaseModel):
    prestructural: int = Field(
        ...,
        ge=0,
        le=10,
        description=(
            "Score (0-10). Score high ONLY if the answer completely misses the point, "
            "uses irrelevant information, or shows deep misunderstanding. "
            "(A highly competent answer should score 0 here)."
        )
    )
    unistructural: int = Field(
        ...,
        ge=0,
        le=10,
        description="Score (0-10) for identifying and correctly focusing on at least one relevant, standalone aspect."
    )
    multistructural: int = Field(
        ...,
        ge=0,
        le=10,
        description="Score (0-10) for identifying multiple relevant aspects accurately, even if treated independently."
    )
    relational: int = Field(
        ...,
        ge=0,
        le=10,
        description="Score (0-10) for integrating various aspects into a cohesive, logically interconnected structure."
    )
    extended_abstract: int = Field(
        ...,
        ge=0,
        le=10,
        description="Score (0-10) for generalizing the integrated structure to propose hypotheses, new domains, or novel theoretical implications."
    )


class HOTSEvaluation(BaseModel):
    content_mastery: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing mastery of the paper's content."
    )
    logical_flow: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing the logical flow of the user's reasoning."
    )
    critical_thinking: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing depth of critical thinking demonstrated."
    )
    academic_language: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing command of academic language."
    )
    cognitive_synthesis: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing ability to synthesize ideas across the paper."
    )
    memory_retention: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing retention of key details from the paper."
    )
    cognitive_stretch: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score (1-10) representing how far the user's thinking was stretched beyond their baseline."
    )


class CognitiveAssessment(BaseModel):
    blind_spots: List[str] = Field(
        ...,
        description="List specific logical flaws, biases, misinterpretations, or overlooked nuances by the user."
    )
    neuro_growth_feedback: str = Field(
        ...,
        description="Provide an intellectually provocative, brutally honest, yet constructive critique. Pinpoint exactly where their reasoning softened and how they should have challenged the text."
    )
    brain_hack_question: str = Field(
        ...,
        description="Formulate one highly sophisticated, complex question based on this paper's deep mechanics to force the user to build new neural connections in their next session."
    )

    international_metrics_evaluated: InternationalMetricsEvaluated
    solo: SOLOTaxonomy = Field(
        ...,
        description="Evaluation of cognitive processing based on the SOLO Taxonomy."
    )
    hots: HOTSEvaluation = Field(
        ...,
        description="Evaluation of Higher-Order Thinking Skills (HOTS) on a 1-10 scale."
    )


class EvaluationORM(Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        Index(
            "ix_evaluations_user_challenge_answer",
            "user_id", "challenge_id", "challenge_paper_id", "answer_id",
        ),
        Index(
            "ix_evaluations_results_gin",
            "results",
            postgresql_using="gin",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    answer_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("answers.id"),
        nullable=False,
    )
    challenge_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("challenges.id"),
        nullable=False,
    )
    challenge_paper_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("challenge_papers.id"),
        nullable=False,
    )
    results: Mapped[Optional[JSONB]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[TIMESTAMP] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[TIMESTAMP] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )