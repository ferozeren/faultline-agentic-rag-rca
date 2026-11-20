"""Structured, evidence-grounded reports for incident investigations."""

from datetime import UTC, datetime
from enum import Enum
from itertools import pairwise

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.schemas.evidence import Evidence, EvidenceClassification, EvidenceSourceType
from app.schemas.findings import (
    AgentObservation,
    EvidenceGap,
    RootCauseFinding,
)
from app.schemas.findings import (
    Recommendation as LegacyRecommendation,
)
from app.schemas.rca import (
    RCAEvidenceClaim,
    RCAFinding,
    RCAHypothesis,
    RCAStatus,
    RCATimelineEvent,
)
from app.schemas.recommendations import (
    RecommendationCategory,
    RecommendationResult,
    RecommendationStatus,
)
from app.schemas.supervisor import EvidenceConflict


class InvestigationStatus(str, Enum):
    """Lifecycle state reported for an investigation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_EVIDENCE = "no_evidence"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ReportStatus(str, Enum):
    """Completion state of the generated report."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class GroundedReportText(BaseModel):
    """Narrative text accompanied by the source IDs supporting its claims."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_references(self) -> "GroundedReportText":
        for field in ("evidence_ids", "finding_ids", "recommendation_ids"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must not contain duplicate IDs.")
        return self


class ReportNarrative(BaseModel):
    """Restricted structured output used only for report narrative fields."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: GroundedReportText
    investigation_outcome: GroundedReportText
    conclusion: GroundedReportText


class ReportExecutiveSummary(BaseModel):
    """Concise incident outcome, without inferred business impact."""

    model_config = ConfigDict(extra="forbid")

    summary: GroundedReportText
    investigation_outcome: GroundedReportText
    root_cause_summary: str | None = None
    root_cause_finding_id: str | None = None
    key_recommendation_ids: list[str] = Field(default_factory=list)
    business_impact: str | None = None


class IncidentOverview(BaseModel):
    """Incident request details and source-linked reported symptoms."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str | None = None
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    affected_service: str | None = None
    affected_services: list[str] = Field(default_factory=list)
    environment: str | None = None
    severity: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    reported_symptoms: list[AgentObservation] = Field(default_factory=list)


class ReportEvidenceReference(Evidence):
    """An unchanged evidence record with explicit report cross-references."""

    linked_finding_ids: list[str] = Field(default_factory=list)
    linked_hypothesis_ids: list[str] = Field(default_factory=list)
    linked_recommendation_ids: list[str] = Field(default_factory=list)


class EvidenceAnalysis(BaseModel):
    """Evidence organized by source and temporal classification."""

    model_config = ConfigDict(extra="forbid")

    historical_rag_evidence_ids: list[str] = Field(default_factory=list)
    application_log_evidence_ids: list[str] = Field(default_factory=list)
    service_metric_evidence_ids: list[str] = Field(default_factory=list)
    prometheus_evidence_ids: list[str] = Field(default_factory=list)
    grafana_evidence_ids: list[str] = Field(default_factory=list)
    verified_facts: list[RCAEvidenceClaim] = Field(default_factory=list)
    hypotheses: list[RCAHypothesis] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    unresolved_conflicts: list[EvidenceConflict] = Field(default_factory=list)


class RootCauseAnalysisSection(BaseModel):
    """Validated RCA and its supporting uncertainty and alternatives."""

    model_config = ConfigDict(extra="forbid")

    status: RCAStatus | None = None
    primary_finding: RCAFinding | None = None
    alternative_hypotheses: list[RCAHypothesis] = Field(default_factory=list)
    verified_facts: list[RCAEvidenceClaim] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    unresolved_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_explanation: str | None = None
    analysis_summary: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise TypeError("confidence must be a number between 0 and 1.")
        return value

    @model_validator(mode="after")
    def validate_primary_finding_status(self) -> "RootCauseAnalysisSection":
        if self.status is RCAStatus.ANALYZED and self.primary_finding is None:
            raise ValueError("analyzed reports require a validated primary finding.")
        if self.status is not RCAStatus.ANALYZED and self.primary_finding is not None:
            raise ValueError("non-analyzed reports cannot assert a primary finding.")
        return self


class RecommendationSections(BaseModel):
    """Recommendation IDs grouped by action category."""

    model_config = ConfigDict(extra="forbid")

    corrective_action_ids: list[str] = Field(default_factory=list)
    preventive_action_ids: list[str] = Field(default_factory=list)
    diagnostic_action_ids: list[str] = Field(default_factory=list)


class UnresolvedIssue(BaseModel):
    """A gap or conflict that remains open at report time."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)


class InvestigationConclusion(BaseModel):
    """Evidence-grounded conclusion and explicit report methodology."""

    model_config = ConfigDict(extra="forbid")

    summary: GroundedReportText
    methodology: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class InvestigationReport(BaseModel):
    """Structured summary of investigation evidence and advisory conclusions."""

    model_config = ConfigDict(extra="forbid")

    # Legacy fields remain stable for existing report consumers.
    incident_id: str | None = None
    incident_summary: str = Field(min_length=1)
    affected_service: str | None = None
    investigation_status: InvestigationStatus
    observations: list[AgentObservation] = Field(default_factory=list)
    historical_findings: list[AgentObservation] = Field(default_factory=list)
    current_findings: list[AgentObservation] = Field(default_factory=list)
    root_cause: RootCauseFinding | None = None
    recommendations: list[LegacyRecommendation] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    report_status: ReportStatus | None = None
    executive_summary: ReportExecutiveSummary | None = None
    incident_overview: IncidentOverview | None = None
    evidence_references: list[ReportEvidenceReference] = Field(default_factory=list)
    evidence_analysis: EvidenceAnalysis | None = None
    timeline: list[RCATimelineEvent] = Field(default_factory=list)
    root_cause_analysis: RootCauseAnalysisSection | None = None
    recommendation_result: RecommendationResult | None = None
    recommendation_sections: RecommendationSections | None = None
    unresolved_issues: list[UnresolvedIssue] = Field(default_factory=list)
    conclusion: InvestigationConclusion | None = None
    missing_sections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_references(self, info: ValidationInfo) -> "InvestigationReport":
        if any(
            finding.classification is not EvidenceClassification.HISTORICAL
            for finding in self.historical_findings
        ):
            raise ValueError(
                "historical_findings must contain historical observations."
            )
        if any(
            finding.classification is not EvidenceClassification.CURRENT
            for finding in self.current_findings
        ):
            raise ValueError("current_findings must contain current observations.")

        context = info.context or {}
        evidence_by_id = context.get("evidence_by_id")
        known_evidence: dict[str, Evidence] = {
            item.evidence_id: Evidence.model_validate(
                item.model_dump(
                    mode="python",
                    exclude={
                        "linked_finding_ids",
                        "linked_hypothesis_ids",
                        "linked_recommendation_ids",
                    },
                )
            )
            for item in self.evidence_references
        }
        if evidence_by_id is not None:
            if not isinstance(evidence_by_id, dict):
                raise ValueError("evidence_by_id context must be a mapping.")
            known_evidence = {}
            for key, candidate in evidence_by_id.items():
                item = Evidence.model_validate(candidate)
                if key != item.evidence_id:
                    raise ValueError("Evidence mapping keys must match evidence IDs.")
                known_evidence[key] = item

        report_reference_ids = [
            item.evidence_id for item in self.evidence_references
        ]
        if len(report_reference_ids) != len(set(report_reference_ids)):
            raise ValueError("Report evidence references must be unique.")

        for reference in self.evidence_references:
            evidence_id = reference.evidence_id
            source = known_evidence.get(evidence_id)
            if source is None:
                raise ValueError(f"Unknown report evidence ID: {evidence_id}.")
            source_evidence = Evidence.model_validate(
                source.model_dump(
                    mode="python",
                    exclude={
                        "linked_finding_ids",
                        "linked_hypothesis_ids",
                        "linked_recommendation_ids",
                    },
                )
                if isinstance(source, Evidence)
                else source
            )
            report_evidence = Evidence.model_validate(
                reference.model_dump(
                    exclude={
                        "linked_finding_ids",
                        "linked_hypothesis_ids",
                        "linked_recommendation_ids",
                    }
                )
            )
            if report_evidence != source_evidence:
                raise ValueError(
                    f"Report evidence {evidence_id!r} does not match its source record."
                )

        finding_ids = set(context.get("finding_ids", []))
        hypothesis_ids = set(context.get("hypothesis_ids", []))
        if "finding_ids" not in context and self.root_cause_analysis:
            finding = self.root_cause_analysis.primary_finding
            finding_ids = {finding.finding_id} if finding else set()
        if "hypothesis_ids" not in context and self.root_cause_analysis:
            hypothesis_ids = {
                item.hypothesis_id
                for item in self.root_cause_analysis.alternative_hypotheses
            }

        recommendation_result = self.recommendation_result
        recommendation_by_id = {
            item.recommendation_id: item
            for item in (
                recommendation_result.recommendations
                if recommendation_result is not None
                else []
            )
        }
        recommendation_ids = set(
            context.get("recommendation_ids", recommendation_by_id)
        )
        expected_finding_links: dict[str, set[str]] = {}
        expected_hypothesis_links: dict[str, set[str]] = {}
        expected_recommendation_links: dict[str, set[str]] = {}
        if self.root_cause_analysis:
            finding = self.root_cause_analysis.primary_finding
            if finding:
                finding_evidence_ids = [
                    *finding.supporting_evidence_ids,
                    *finding.contradicting_evidence_ids,
                    *(
                        evidence_id
                        for claim in (
                            *finding.contributing_factors,
                            *finding.downstream_impact,
                            *((finding.triggering_event,) if finding.triggering_event else ()),
                        )
                        for evidence_id in claim.evidence_ids
                    ),
                ]
                for evidence_id in finding_evidence_ids:
                    expected_finding_links.setdefault(evidence_id, set()).add(
                        finding.finding_id
                    )
            for hypothesis in self.root_cause_analysis.alternative_hypotheses:
                for evidence_id in (
                    *hypothesis.supporting_evidence_ids,
                    *hypothesis.contradicting_evidence_ids,
                ):
                    expected_hypothesis_links.setdefault(evidence_id, set()).add(
                        hypothesis.hypothesis_id
                    )
        if recommendation_result:
            for recommendation in recommendation_result.recommendations:
                for evidence_id in recommendation.supporting_evidence_ids:
                    expected_recommendation_links.setdefault(evidence_id, set()).add(
                        recommendation.recommendation_id
                    )

        def validate_evidence_ids(values: list[str], field: str) -> None:
            unknown = set(values) - set(known_evidence)
            if unknown:
                raise ValueError(
                    f"{field} references unknown evidence IDs: "
                    f"{', '.join(sorted(unknown))}."
                )

        def validate_finding_ids(values: list[str], field: str) -> None:
            unknown = set(values) - finding_ids
            if unknown:
                raise ValueError(
                    f"{field} references unknown finding IDs: "
                    f"{', '.join(sorted(unknown))}."
                )

        def validate_hypothesis_ids(values: list[str], field: str) -> None:
            unknown = set(values) - hypothesis_ids
            if unknown:
                raise ValueError(
                    f"{field} references unknown hypothesis IDs: "
                    f"{', '.join(sorted(unknown))}."
                )

        def validate_text_references(text: GroundedReportText, field: str) -> None:
            validate_evidence_ids(text.evidence_ids, field)
            validate_finding_ids(text.finding_ids, field)
            unknown_recommendations = set(text.recommendation_ids) - recommendation_ids
            if unknown_recommendations:
                raise ValueError(
                    f"{field} references unknown recommendation IDs: "
                    f"{', '.join(sorted(unknown_recommendations))}."
                )
            if (
                self.root_cause_analysis
                and self.root_cause_analysis.primary_finding
                and self.root_cause_analysis.primary_finding.finding_id
                in text.finding_ids
                and not any(
                    known_evidence[evidence_id].classification
                    is EvidenceClassification.CURRENT
                    for evidence_id in text.evidence_ids
                )
            ):
                raise ValueError(
                    f"{field} must cite current evidence when stating a root cause."
                )

        if self.evidence_analysis:
            source_groups = {
                "historical_rag_evidence_ids": (
                    EvidenceSourceType.HISTORICAL_RAG,
                    EvidenceClassification.HISTORICAL,
                ),
                "application_log_evidence_ids": (
                    EvidenceSourceType.APPLICATION_LOGS,
                    EvidenceClassification.CURRENT,
                ),
                "service_metric_evidence_ids": (
                    EvidenceSourceType.SERVICE_METRICS,
                    EvidenceClassification.CURRENT,
                ),
                "prometheus_evidence_ids": (
                    EvidenceSourceType.PROMETHEUS,
                    EvidenceClassification.CURRENT,
                ),
                "grafana_evidence_ids": (
                    EvidenceSourceType.GRAFANA,
                    EvidenceClassification.CURRENT,
                ),
            }
            grouped_ids: list[str] = []
            for field, (source_type, classification) in source_groups.items():
                values = getattr(self.evidence_analysis, field)
                if len(values) != len(set(values)):
                    raise ValueError(f"{field} must not contain duplicate IDs.")
                validate_evidence_ids(values, field)
                for evidence_id in values:
                    item = known_evidence[evidence_id]
                    if (
                        item.source_type is not source_type
                        or item.classification is not classification
                    ):
                        raise ValueError(
                            f"{field} contains evidence with an incorrect source "
                            "or classification."
                        )
                grouped_ids.extend(values)
            if set(grouped_ids) != set(report_reference_ids):
                raise ValueError(
                    "Evidence source groups must account for every report reference."
                )
            for claim in self.evidence_analysis.verified_facts:
                validate_evidence_ids(claim.evidence_ids, "verified_facts")
                if any(
                    known_evidence[item].classification is not claim.classification
                    for item in claim.evidence_ids
                ):
                    raise ValueError(
                        "Verified-fact classification must match cited evidence."
                    )
            for hypothesis in self.evidence_analysis.hypotheses:
                validate_hypothesis_ids(
                    [hypothesis.hypothesis_id], "evidence_analysis.hypotheses"
                )
                validate_evidence_ids(
                    [
                        *hypothesis.supporting_evidence_ids,
                        *hypothesis.contradicting_evidence_ids,
                    ],
                    "evidence_analysis.hypotheses",
                )
            validate_evidence_ids(
                self.evidence_analysis.contradictory_evidence_ids,
                "contradictory_evidence_ids",
            )

        for reference in self.evidence_references:
            link_fields = (
                reference.linked_finding_ids,
                reference.linked_hypothesis_ids,
                reference.linked_recommendation_ids,
            )
            if any(len(values) != len(set(values)) for values in link_fields):
                raise ValueError("Evidence link IDs must be unique.")
            validate_finding_ids(
                reference.linked_finding_ids,
                "evidence_references.linked_finding_ids",
            )
            validate_hypothesis_ids(
                reference.linked_hypothesis_ids,
                "evidence_references.linked_hypothesis_ids",
            )
            unknown_recommendations = (
                set(reference.linked_recommendation_ids) - recommendation_ids
            )
            if unknown_recommendations:
                raise ValueError(
                    "evidence_references.linked_recommendation_ids contains "
                    "unknown recommendation IDs."
                )
            if (
                set(reference.linked_finding_ids)
                != expected_finding_links.get(reference.evidence_id, set())
                or set(reference.linked_hypothesis_ids)
                != expected_hypothesis_links.get(reference.evidence_id, set())
                or set(reference.linked_recommendation_ids)
                != expected_recommendation_links.get(reference.evidence_id, set())
            ):
                raise ValueError(
                    "Evidence links must match the validated RCA and recommendations."
                )

        for event in self.timeline:
            validate_evidence_ids(event.evidence_ids, "timeline")
            source_records = [known_evidence[item] for item in event.evidence_ids]
            if any(item.source_type is not event.source for item in source_records):
                raise ValueError("Timeline source must match referenced evidence.")
            if event.service and any(
                item.service and item.service != event.service
                for item in source_records
            ):
                raise ValueError("Timeline service must match referenced evidence.")
            if event.timestamp is not None:
                available_timestamps = {
                    timestamp
                    for item in source_records
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
        event_ids = [event.event_id for event in self.timeline]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Timeline event IDs must be unique.")
        timestamps = [
            event.timestamp for event in self.timeline if event.timestamp is not None
        ]
        for previous, current in pairwise(timestamps):
            if (previous.tzinfo is None) != (current.tzinfo is None):
                continue
            if previous > current:
                raise ValueError("Timeline events must be in chronological order.")

        traceability_requested = (
            evidence_by_id is not None
            or bool(self.evidence_references)
            or self.report_status is not None
        )
        if traceability_requested:
            observations = [
                *self.observations,
                *self.historical_findings,
                *self.current_findings,
                *(
                    self.incident_overview.reported_symptoms
                    if self.incident_overview
                    else []
                ),
            ]
            for observation in observations:
                validate_evidence_ids(
                    observation.evidence_ids,
                    f"observation {observation.observation_id!r}",
                )
                cited = [known_evidence[item] for item in observation.evidence_ids]
                if any(
                    item.classification is not observation.classification
                    for item in cited
                ) or (
                    observation.source_types
                    and any(item.source_type not in observation.source_types for item in cited)
                ):
                    raise ValueError(
                        f"Observation {observation.observation_id!r} does not match "
                        "its cited evidence."
                    )
            for recommendation in self.recommendations:
                validate_evidence_ids(
                    recommendation.supporting_evidence_ids,
                    "legacy recommendations",
                )
            if self.root_cause:
                validate_evidence_ids(
                    [
                        *self.root_cause.supporting_evidence_ids,
                        *self.root_cause.contradicting_evidence_ids,
                    ],
                    "legacy root cause",
                )
                if any(
                    known_evidence[item].classification
                    is not EvidenceClassification.CURRENT
                    for item in self.root_cause.supporting_evidence_ids
                ):
                    raise ValueError(
                        "Legacy root causes must use current operational evidence."
                    )

        if self.root_cause_analysis:
            section = self.root_cause_analysis
            if section.primary_finding:
                finding = section.primary_finding
                validate_finding_ids([finding.finding_id], "primary_finding")
                current_support = set(finding.supporting_evidence_ids)
                validate_evidence_ids(
                    [
                        *finding.supporting_evidence_ids,
                        *finding.contradicting_evidence_ids,
                    ],
                    "primary_finding",
                )
                if not current_support or any(
                    known_evidence[item].classification
                    is not EvidenceClassification.CURRENT
                    for item in current_support
                ):
                    raise ValueError(
                        "Primary findings must use current operational evidence."
                    )
                for claim in (
                    *finding.contributing_factors,
                    *finding.downstream_impact,
                    *((finding.triggering_event,) if finding.triggering_event else ()),
                ):
                    validate_evidence_ids(claim.evidence_ids, "primary_finding claims")
                    if any(
                        known_evidence[item].classification
                        is not claim.classification
                        for item in claim.evidence_ids
                    ):
                        raise ValueError(
                            "Primary finding claims must match evidence classification."
                        )
            for hypothesis in section.alternative_hypotheses:
                validate_hypothesis_ids(
                    [hypothesis.hypothesis_id], "alternative_hypotheses"
                )
                validate_evidence_ids(
                    [
                        *hypothesis.supporting_evidence_ids,
                        *hypothesis.contradicting_evidence_ids,
                    ],
                    "alternative_hypotheses",
                )
            for claim in section.verified_facts:
                validate_evidence_ids(claim.evidence_ids, "root_cause_analysis")
                if any(
                    known_evidence[item].classification is not claim.classification
                    for item in claim.evidence_ids
                ):
                    raise ValueError(
                        "RCA verified-fact classification must match evidence."
                    )

        for text, field in (
            (
                self.executive_summary.summary,
                "executive_summary.summary",
            )
            if self.executive_summary
            else (None, ""),
            (
                self.executive_summary.investigation_outcome,
                "executive_summary.investigation_outcome",
            )
            if self.executive_summary
            else (None, ""),
            (self.conclusion.summary, "conclusion.summary")
            if self.conclusion
            else (None, ""),
        ):
            if text is not None:
                validate_text_references(text, field)

        if self.executive_summary:
            summary = self.executive_summary
            if summary.root_cause_finding_id:
                validate_finding_ids(
                    [summary.root_cause_finding_id],
                    "executive_summary.root_cause_finding_id",
                )
            unknown_recommendations = (
                set(summary.key_recommendation_ids) - recommendation_ids
            )
            if unknown_recommendations:
                raise ValueError(
                    "Executive summary references unknown recommendation IDs."
                )
            if len(summary.key_recommendation_ids) != len(
                set(summary.key_recommendation_ids)
            ):
                raise ValueError("Executive-summary recommendation IDs must be unique.")
            if summary.root_cause_summary is not None:
                finding = (
                    self.root_cause_analysis.primary_finding
                    if self.root_cause_analysis
                    else None
                )
                if (
                    finding is None
                    or summary.root_cause_finding_id != finding.finding_id
                    or summary.root_cause_summary != finding.summary
                ):
                    raise ValueError(
                        "Executive root-cause summary must match the validated finding."
                    )
            if summary.business_impact is not None and not any(
                summary.business_impact in item.description
                for item in known_evidence.values()
                if item.classification is EvidenceClassification.CURRENT
            ):
                raise ValueError(
                    "Business impact must be directly supported by current evidence."
                )

        if self.recommendation_sections:
            sections = self.recommendation_sections
            categorized_ids = {
                RecommendationCategory.CORRECTIVE: sections.corrective_action_ids,
                RecommendationCategory.PREVENTIVE: sections.preventive_action_ids,
                RecommendationCategory.DIAGNOSTIC: sections.diagnostic_action_ids,
            }
            flattened = [
                recommendation_id
                for values in categorized_ids.values()
                for recommendation_id in values
            ]
            if len(flattened) != len(set(flattened)):
                raise ValueError("Recommendation section IDs must be unique.")
            if set(flattened) != set(recommendation_by_id):
                raise ValueError(
                    "Recommendation sections must account for every recommendation."
                )
            for category, values in categorized_ids.items():
                if any(
                    recommendation_by_id[item].category is not category
                    for item in values
                ):
                    raise ValueError(
                        "Recommendation section does not match action category."
                    )

        if recommendation_result:
            if recommendation_result.status in {
                RecommendationStatus.FAILED,
                RecommendationStatus.SKIPPED,
                RecommendationStatus.INSUFFICIENT_EVIDENCE,
            } and recommendation_result.recommendations:
                raise ValueError(
                    "Non-actionable recommendation results cannot contain actions."
                )
            if recommendation_result.status is RecommendationStatus.COMPLETED and {
                item.category for item in recommendation_result.recommendations
            } != set(RecommendationCategory):
                raise ValueError(
                    "Completed recommendation results require every action category."
                )
            allowed_finding_ids = finding_ids
            primary_support = (
                set(self.root_cause_analysis.primary_finding.supporting_evidence_ids)
                if self.root_cause_analysis
                and self.root_cause_analysis.primary_finding
                else set()
            )
            for recommendation in recommendation_result.recommendations:
                validate_evidence_ids(
                    recommendation.supporting_evidence_ids,
                    "recommendation_result",
                )
                validate_finding_ids(
                    recommendation.related_finding_ids,
                    "recommendation_result",
                )
                if any(
                    known_evidence[item].classification
                    is not EvidenceClassification.CURRENT
                    for item in recommendation.supporting_evidence_ids
                ):
                    raise ValueError(
                        "Recommendations must use current operational evidence."
                    )
                if (
                    not set(recommendation.supporting_evidence_ids).issubset(
                        primary_support
                    )
                    or not set(recommendation.related_finding_ids).issubset(
                        allowed_finding_ids
                    )
                ):
                    raise ValueError(
                        "Recommendations must be supported by the validated primary RCA."
                    )
            validate_finding_ids(
                recommendation_result.source_rca_finding_ids,
                "recommendation_result.source_rca_finding_ids",
            )
            if self.report_status is ReportStatus.COMPLETED:
                categories = {
                    item.category for item in recommendation_result.recommendations
                }
                if (
                    recommendation_result.status is not RecommendationStatus.COMPLETED
                    or categories != set(RecommendationCategory)
                ):
                    raise ValueError(
                        "Completed reports require completed recommendations from "
                        "all recommendation categories."
                    )

        if (
            self.report_status is ReportStatus.COMPLETED
            and (
                self.root_cause_analysis is None
                or self.root_cause_analysis.status is not RCAStatus.ANALYZED
                or self.root_cause_analysis.primary_finding is None
                or recommendation_result is None
            )
        ):
            raise ValueError(
                "Completed reports require a validated RCA and recommendations."
            )
        if (
            self.report_status is ReportStatus.INSUFFICIENT_EVIDENCE
            and self.root_cause_analysis
            and self.root_cause_analysis.primary_finding is not None
        ):
            raise ValueError(
                "Insufficient-evidence reports cannot assert a primary finding."
            )

        for issue in self.unresolved_issues:
            validate_evidence_ids(issue.evidence_ids, "unresolved_issues")
            validate_finding_ids(issue.finding_ids, "unresolved_issues")
        return self
