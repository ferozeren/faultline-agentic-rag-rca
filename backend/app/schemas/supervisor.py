"""Structured, evidence-referenced decisions made before root-cause analysis."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.schemas.evidence import EvidenceClassification, EvidenceSourceType
from app.schemas.findings import EvidenceGap


class DecisionStatus(str, Enum):
    """Outcome of the evidence-readiness assessment."""

    READY_FOR_RCA = "ready_for_rca"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    NO_EVIDENCE = "no_evidence"


class EvidenceSufficiency(str, Enum):
    """How well the evidence supports an RCA investigation."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class RecommendedNextStage(str, Enum):
    """The next permitted investigation stage."""

    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    ADDITIONAL_EVIDENCE_COLLECTION = "additional_evidence_collection"
    EVIDENCE_CONFLICT_REVIEW = "evidence_conflict_review"
    MANUAL_REVIEW = "manual_review"


class HistoricalCurrentAlignment(str, Enum):
    """Relationship between historical references and current observations."""

    ALIGNED = "aligned"
    DISAGREE = "disagree"
    NOT_COMPARABLE = "not_comparable"
    NO_HISTORICAL = "no_historical"
    NO_CURRENT = "no_current"


class SourceAssessmentStatus(str, Enum):
    """Observed collection result for one specialist source."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class ConflictType(str, Enum):
    """Evidence relationship requiring reconciliation."""

    CURRENT_CURRENT = "current_current"
    HISTORICAL_CURRENT = "historical_current"


class EvidenceStatement(BaseModel):
    """A fact transcribed from a specific evidence record."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)


class InvestigationEvidenceSummary(BaseModel):
    """Keep observed facts, historical context, and hypotheses distinct."""

    model_config = ConfigDict(extra="forbid")

    observed_facts: list[EvidenceStatement] = Field(default_factory=list)
    historical_context: list[EvidenceStatement] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)


class SourceAssessment(BaseModel):
    """Evidence collected or unavailable from a single source."""

    model_config = ConfigDict(extra="forbid")

    source_type: EvidenceSourceType
    classification: EvidenceClassification
    status: SourceAssessmentStatus
    evidence_count: int = Field(ge=0)
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_classification_and_count(self) -> "SourceAssessment":
        expected_classification = (
            EvidenceClassification.HISTORICAL
            if self.source_type is EvidenceSourceType.HISTORICAL_RAG
            else EvidenceClassification.CURRENT
        )
        if self.classification is not expected_classification:
            raise ValueError("Source assessment classification does not match source.")
        if self.evidence_count != len(self.evidence_ids):
            raise ValueError("evidence_count must match evidence_ids.")
        if self.status is SourceAssessmentStatus.NO_EVIDENCE and self.evidence_ids:
            raise ValueError("A no-evidence source assessment cannot list evidence IDs.")
        if self.status is SourceAssessmentStatus.AVAILABLE and not self.evidence_ids:
            raise ValueError("An available source assessment requires evidence IDs.")
        return self


class EvidenceConflict(BaseModel):
    """A conflict candidate that must be reviewed rather than silently resolved."""

    model_config = ConfigDict(extra="forbid")

    conflict_type: ConflictType
    evidence_ids: list[str] = Field(min_length=2)
    description: str = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def require_distinct_evidence_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Conflicting evidence IDs must be distinct.")
        return value


class InvestigationDecision(BaseModel):
    """Validated decision on whether evidence is ready for an RCA investigation."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str | None = None
    decision_status: DecisionStatus
    evidence_sufficiency: EvidenceSufficiency
    evidence_summary: InvestigationEvidenceSummary
    source_assessments: list[SourceAssessment] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    historical_current_alignment: HistoricalCurrentAlignment
    confidence: float = Field(ge=0, le=1)
    decision_rationale: str = Field(min_length=1)
    recommended_next_stage: RecommendedNextStage
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise TypeError("confidence must be a number between 0 and 1.")
        return value

    @model_validator(mode="after")
    def validate_decision_consistency_and_evidence_ids(
        self,
        info: ValidationInfo,
    ) -> "InvestigationDecision":
        ids_in_assessments = {
            evidence_id
            for assessment in self.source_assessments
            for evidence_id in assessment.evidence_ids
        }
        ids_in_summary = {
            statement.evidence_id
            for statement in (
                *self.evidence_summary.observed_facts,
                *self.evidence_summary.historical_context,
            )
        }
        all_referenced_ids = (
            ids_in_assessments
            | ids_in_summary
            | set(self.supporting_evidence_ids)
            | {
                evidence_id
                for conflict in self.conflicts
                for evidence_id in conflict.evidence_ids
            }
        )
        available_ids = (info.context or {}).get("available_evidence_ids")
        if available_ids is not None:
            unknown_ids = all_referenced_ids - set(available_ids)
            if unknown_ids:
                raise ValueError(
                    "Decision references evidence IDs not present in the input: "
                    f"{', '.join(sorted(unknown_ids))}."
                )

        if not set(self.supporting_evidence_ids).issubset(ids_in_assessments):
            raise ValueError(
                "Supporting evidence IDs must appear in source assessments."
            )
        if not ids_in_summary.issubset(ids_in_assessments):
            raise ValueError(
                "Evidence summary IDs must appear in source assessments."
            )
        conflict_ids = {
            evidence_id
            for conflict in self.conflicts
            for evidence_id in conflict.evidence_ids
        }
        if not conflict_ids.issubset(ids_in_assessments):
            raise ValueError("Conflict IDs must appear in source assessments.")
        if self.decision_status is DecisionStatus.NO_EVIDENCE and (
            self.evidence_sufficiency is not EvidenceSufficiency.INSUFFICIENT
            or self.recommended_next_stage is not RecommendedNextStage.MANUAL_REVIEW
            or self.supporting_evidence_ids
        ):
            raise ValueError("no_evidence decisions must request manual review.")
        if self.decision_status is DecisionStatus.CONFLICTING_EVIDENCE and (
            self.recommended_next_stage
            is not RecommendedNextStage.EVIDENCE_CONFLICT_REVIEW
            or not self.conflicts
        ):
            raise ValueError(
                "conflicting_evidence decisions require conflicts and conflict review."
            )
        if self.decision_status is DecisionStatus.READY_FOR_RCA and (
            self.evidence_sufficiency is not EvidenceSufficiency.SUFFICIENT
            or self.recommended_next_stage
            is not RecommendedNextStage.ROOT_CAUSE_ANALYSIS
        ):
            raise ValueError(
                "ready_for_rca decisions require sufficient evidence and RCA as next stage."
            )
        if self.decision_status is DecisionStatus.NEEDS_MORE_EVIDENCE and (
            self.evidence_sufficiency is EvidenceSufficiency.SUFFICIENT
            or self.recommended_next_stage
            is not RecommendedNextStage.ADDITIONAL_EVIDENCE_COLLECTION
        ):
            raise ValueError(
                "needs_more_evidence decisions require additional evidence collection."
            )
        return self


class SupervisorModelAssessment(BaseModel):
    """Restricted model output; it cannot provide causes or arbitrary claims."""

    model_config = ConfigDict(extra="forbid")

    evidence_sufficiency: EvidenceSufficiency
    historical_current_alignment: HistoricalCurrentAlignment
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise TypeError("confidence must be a number between 0 and 1.")
        return value

    @field_validator("conflicting_evidence_ids")
    @classmethod
    def require_distinct_conflict_ids(cls, value: list[str]) -> list[str]:
        if value and len(set(value)) < 2:
            raise ValueError("A conflict requires at least two distinct evidence IDs.")
        return value

    @model_validator(mode="after")
    def validate_evidence_ids(self, info: ValidationInfo) -> "SupervisorModelAssessment":
        available_ids = (info.context or {}).get("available_evidence_ids")
        if available_ids is not None:
            referenced = set(self.conflicting_evidence_ids) | set(
                self.supporting_evidence_ids
            )
            unknown_ids = referenced - set(available_ids)
            if unknown_ids:
                raise ValueError(
                    "Model assessment references unknown evidence IDs: "
                    f"{', '.join(sorted(unknown_ids))}."
                )
        return self
