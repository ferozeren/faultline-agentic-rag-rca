"""Structured recommendation output for incident remediation and prevention."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RecommendationCategory(str, Enum):
    """Type of recommendation."""

    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    DIAGNOSTIC = "diagnostic"


class RecommendationPriority(str, Enum):
    """Operational priority for a recommendation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationUrgency(str, Enum):
    """How quickly the recommendation should be acted on."""

    IMMEDIATE = "immediate"
    WITHIN_24_HOURS = "within_24_hours"
    PLANNED = "planned"
    INFORMATIONAL = "informational"


class RecommendationStatus(str, Enum):
    """Outcome of recommendation generation for a validated RCA."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class Recommendation(BaseModel):
    """Evidence-informed corrective, preventive, or diagnostic action."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: RecommendationCategory
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    urgency: RecommendationUrgency = RecommendationUrgency.WITHIN_24_HOURS
    rationale: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    related_finding_ids: list[str] = Field(min_length=1)
    affected_services: list[str] = Field(default_factory=list)
    expected_impact: str | None = None
    implementation_risk: str | None = None
    implementation_effort: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(min_length=1)
    rollback_considerations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    requires_human_approval: bool = False

    @field_validator(
        "recommendation_id",
        "title",
        "description",
        "rationale",
        mode="before",
    )
    @classmethod
    def reject_blank_text(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Text fields must not be blank.")
        return value

    @field_validator(
        "supporting_evidence_ids",
        "related_finding_ids",
        "verification_steps",
        mode="before",
    )
    @classmethod
    def reject_blank_list_items(cls, value: object) -> object:
        if isinstance(value, list) and any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("List items must not be blank.")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps this as ValidationError.
                "confidence must be a number between 0 and 1."
            )
        return value


class RecommendationBatch(BaseModel):
    """Structured model output before deterministic RCA-reference validation."""

    model_config = ConfigDict(extra="forbid")

    recommendations: list[Recommendation] = Field(min_length=3)

    @model_validator(mode="after")
    def require_all_recommendation_categories(self) -> RecommendationBatch:
        categories = {
            recommendation.category for recommendation in self.recommendations
        }
        if categories != set(RecommendationCategory):
            raise ValueError(
                "Recommendation output must include corrective, preventive, "
                "and diagnostic actions."
            )
        recommendation_ids = [
            recommendation.recommendation_id for recommendation in self.recommendations
        ]
        if len(recommendation_ids) != len(set(recommendation_ids)):
            raise ValueError("Recommendation identifiers must be unique.")
        return self


class RecommendationResult(BaseModel):
    """Derived recommendations produced from a validated RCA result."""

    model_config = ConfigDict(extra="forbid")

    status: RecommendationStatus
    incident_id: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    corrective_action_count: int = 0
    preventive_action_count: int = 0
    diagnostic_action_count: int = 0
    overall_summary: str = Field(min_length=1)
    source_rca_finding_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_counts(self) -> RecommendationResult:
        expected = {
            RecommendationCategory.CORRECTIVE: self.corrective_action_count,
            RecommendationCategory.PREVENTIVE: self.preventive_action_count,
            RecommendationCategory.DIAGNOSTIC: self.diagnostic_action_count,
        }
        actual = {
            category: sum(
                recommendation.category is category
                for recommendation in self.recommendations
            )
            for category in RecommendationCategory
        }
        if expected != actual:
            raise ValueError("Recommendation action counts must match the actions.")
        recommendation_ids = [
            recommendation.recommendation_id for recommendation in self.recommendations
        ]
        if len(recommendation_ids) != len(set(recommendation_ids)):
            raise ValueError("Recommendation identifiers must be unique.")
        return self
