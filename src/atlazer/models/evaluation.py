from pydantic import BaseModel, Field
from typing import List


class InternationalMetricsEvaluated(BaseModel):
    issue_clarification_depth: str = Field(
        ...,
        description="Evaluate the sharpness in unraveling complex issues within the paper."
    )
    evidence_interrogativity: str = Field(
        ...,
        description="Evaluate whether the user aggressively questioned the data/methodologies rather than passively summarizing."
    )
    assumption_and_bias_detection: str = Field(
        ...,
        description="Evaluate the ability to detect the author's underlying assumptions and the user's own cognitive biases."
    )
    divergent_synthesis: str = Field(
        ...,
        description="Evaluate the capacity to bridge fragmented pieces of information into a novel, overarching perspective."
    )
    systemic_implication: str = Field(
        ...,
        description="Evaluate the foresight in predicting long-term, real-world impacts or consequences of the paper's findings."
    )


class CognitiveAssessment(BaseModel):
    content_mastery_score: int = Field(..., description="Score representing mastery of the paper's content.")
    logical_flow_score: int = Field(..., description="Score representing the logical flow of the user's reasoning.")
    critical_thinking_score: int = Field(..., description="Score representing depth of critical thinking demonstrated.")
    academic_language_score: int = Field(..., description="Score representing command of academic language.")
    cognitive_synthesis_score: int = Field(..., description="Score representing ability to synthesize ideas across the paper.")
    memory_retention_score: int = Field(..., description="Score representing retention of key details from the paper.")
    cognitive_stretch_score: int = Field(..., description="Score representing how far the user's thinking was stretched beyond their baseline.")

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