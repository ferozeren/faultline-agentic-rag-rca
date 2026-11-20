"""Structured observations, findings, recommendations, and evidence gaps."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.evidence import EvidenceClassification, EvidenceSourceType


class AgentObservation(BaseModel):
    """A source-attributed observation made during investigation."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    source_types: list[EvidenceSourceType] = Field(default_factory=list)
    classification: EvidenceClassification


class RootCauseFinding(BaseModel):
    """A candidate cause supported by linked evidence."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    probable_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    alternative_causes: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps ValueError as ValidationError.
                "confidence must be a number between 0 and 1."
            )
        return value


class RecommendationPriority(str, Enum):
    """Relative urgency for advisory recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(BaseModel):
    """An advisory action; this schema does not execute remediation."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    priority: RecommendationPriority
    expected_outcome: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    risk_or_caution: str | None = None
    advisory_only: Literal[True] = True


class EvidenceGap(BaseModel):
    """A material absence or limitation in the evidence collected."""

    model_config = ConfigDict(extra="forbid")

    gap_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    missing_source_types: list[EvidenceSourceType] = Field(default_factory=list)
    impact: str | None = None
    recommended_data_sources: list[str] = Field(default_factory=list)
