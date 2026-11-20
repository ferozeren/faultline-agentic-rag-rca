"""Typed, evidence-referenced output for root-cause analysis."""

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

from app.schemas.evidence import Evidence, EvidenceClassification, EvidenceSourceType
from app.schemas.findings import EvidenceGap
from app.schemas.supervisor import EvidenceConflict


class RCAStatus(str, Enum):
    """Outcome of an RCA analysis attempt."""

    ANALYZED = "analyzed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RCAHypothesisStatus(str, Enum):
    """Evidence posture of an alternative causal explanation."""

    PLAUSIBLE = "plausible"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class RCAEventType(str, Enum):
    """Event category for a source-grounded timeline entry."""

    SYMPTOM = "symptom"
    ERROR = "error"
    ANOMALY = "anomaly"
    DEGRADATION = "degradation"
    DOWNSTREAM_IMPACT = "downstream_impact"
    RECOVERY = "recovery"
    OTHER = "other"


class RCAEvidenceClaim(BaseModel):
    """A claim with source evidence and explicit temporal classification."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    classification: EvidenceClassification


class RCAFinding(BaseModel):
    """A probable cause and its evidence-supported reasoning."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    affected_services: list[str] = Field(default_factory=list)
    contributing_factors: list[RCAEvidenceClaim] = Field(default_factory=list)
    triggering_event: RCAEvidenceClaim | None = None
    downstream_impact: list[RCAEvidenceClaim] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    confidence_explanation: str = Field(min_length=1)
    causal_reasoning: str = Field(min_length=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps this as ValidationError.
                "confidence must be a number between 0 and 1."
            )
        return value


class RCAHypothesis(BaseModel):
    """An alternative explanation kept separate from verified observations."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    status: RCAHypothesisStatus

    @model_validator(mode="after")
    def require_evidence_reference(self) -> "RCAHypothesis":
        if not self.supporting_evidence_ids and not self.contradicting_evidence_ids:
            raise ValueError(
                "RCA hypotheses must reference supporting or contradicting evidence."
            )
        return self

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps this as ValidationError.
                "confidence must be a number between 0 and 1."
            )
        return value


class RCATimelineEvent(BaseModel):
    """A chronological source event; timestamp may be unknown."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    service: str | None = None
    source: EvidenceSourceType
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    event_type: RCAEventType


class RCAResult(BaseModel):
    """A validated analysis that keeps evidence, inference, and uncertainty distinct."""

    model_config = ConfigDict(extra="forbid")

    status: RCAStatus
    incident_summary: str = Field(min_length=1)
    primary_finding: RCAFinding | None = None
    alternative_hypotheses: list[RCAHypothesis] = Field(default_factory=list)
    timeline: list[RCATimelineEvent] = Field(default_factory=list)
    verified_facts: list[RCAEvidenceClaim] = Field(default_factory=list)
    hypotheses: list[RCAHypothesis] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    unresolved_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)
    confidence_explanation: str = Field(min_length=1)
    analysis_summary: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps this as ValidationError.
                "confidence must be a number between 0 and 1."
            )
        return value

    @model_validator(mode="after")
    def validate_result_and_evidence_references(
        self,
        info: ValidationInfo,
    ) -> "RCAResult":
        if self.status is RCAStatus.ANALYZED and self.primary_finding is None:
            raise ValueError("analyzed results require a primary root-cause finding.")
        if self.status is not RCAStatus.ANALYZED and self.primary_finding is not None:
            raise ValueError("non-analyzed results cannot assert a primary root cause.")

        evidence_by_id = (info.context or {}).get("evidence_by_id")
        if evidence_by_id is None:
            return self
        if not isinstance(evidence_by_id, dict):
            raise ValueError(  # noqa: TRY004 - Invalid model context is a validation error.
                "evidence_by_id validation context must be a mapping."
            )
        evidence_by_id = {
            key: value for key, value in evidence_by_id.items() if isinstance(key, str)
        }

        def validate_ids(evidence_ids: list[str]) -> list[Evidence]:
            missing = set(evidence_ids) - set(evidence_by_id)
            if missing:
                raise ValueError(
                    "RCA references unknown evidence IDs: "
                    f"{', '.join(sorted(missing))}."
                )
            evidence: list[Evidence] = []
            for evidence_id in evidence_ids:
                candidate = evidence_by_id[evidence_id]
                item = (
                    candidate
                    if isinstance(candidate, Evidence)
                    else Evidence.model_validate(candidate)
                )
                if item.evidence_id != evidence_id:
                    raise ValueError("Evidence mapping keys must match evidence IDs.")
                evidence.append(item)
            return evidence

        def validate_claim(claim: RCAEvidenceClaim) -> None:
            records = validate_ids(claim.evidence_ids)
            if any(item.classification is not claim.classification for item in records):
                raise ValueError(
                    "Claim classification must match all referenced evidence."
                )

        for claim in self.verified_facts:
            validate_claim(claim)
        for event in self.timeline:
            records = validate_ids(event.evidence_ids)
            if any(item.source_type is not event.source for item in records):
                raise ValueError("Timeline source must match referenced evidence.")
            if event.service and any(
                item.service and item.service != event.service for item in records
            ):
                raise ValueError("Timeline service must match referenced evidence.")
            if event.timestamp is not None:
                available_timestamps = {
                    timestamp
                    for item in records
                    for timestamp in (
                        item.observed_at,
                        item.time_range_start,
                        item.time_range_end,
                    )
                    if timestamp is not None
                }
                if event.timestamp not in available_timestamps:
                    raise ValueError(
                        "Timeline timestamps must come from referenced evidence."
                    )

        finding = self.primary_finding
        if finding is not None:
            support = validate_ids(finding.supporting_evidence_ids)
            if not support or any(
                item.classification is not EvidenceClassification.CURRENT
                for item in support
            ):
                raise ValueError(
                    "A primary root cause must be supported by current operational evidence only."
                )
            allowed_services = {
                item.service for item in support if item.service is not None
            }
            incident_service = (info.context or {}).get("incident_service")
            if isinstance(incident_service, str) and incident_service:
                allowed_services.add(incident_service)
            if not set(finding.affected_services).issubset(allowed_services):
                raise ValueError(
                    "Affected services must match incident context or supporting evidence."
                )
            validate_ids(finding.contradicting_evidence_ids)
            for claim in (
                *finding.contributing_factors,
                *finding.downstream_impact,
                *((finding.triggering_event,) if finding.triggering_event else ()),
            ):
                validate_claim(claim)
                if claim.classification is not EvidenceClassification.CURRENT:
                    raise ValueError(
                        "Primary finding factors must use current operational evidence."
                    )

        for hypothesis in (*self.alternative_hypotheses, *self.hypotheses):
            validate_ids(hypothesis.supporting_evidence_ids)
            validate_ids(hypothesis.contradicting_evidence_ids)
        for conflict in self.unresolved_conflicts:
            validate_ids(conflict.evidence_ids)
        return self
