"""Validation tests for the structured investigation report."""

from datetime import UTC, datetime, timedelta

import pytest
from app.schemas import (
    Evidence,
    EvidenceAnalysis,
    EvidenceClassification,
    EvidenceSourceType,
    GroundedReportText,
    IncidentOverview,
    InvestigationConclusion,
    InvestigationReport,
    InvestigationStatus,
    RCAEvidenceClaim,
    RCAFinding,
    RCAStatus,
    RCATimelineEvent,
    RecommendationCategory,
    RecommendationResult,
    RecommendationSections,
    RecommendationStatus,
    ReportEvidenceReference,
    ReportExecutiveSummary,
    ReportStatus,
    RootCauseAnalysisSection,
)
from pydantic import ValidationError

_TIME = datetime(2026, 12, 1, 10, 0, tzinfo=UTC)


def _evidence(evidence_id="logs:one", *, timestamp=_TIME, historical=False):
    source_type = (
        EvidenceSourceType.HISTORICAL_RAG
        if historical
        else EvidenceSourceType.APPLICATION_LOGS
    )
    classification = (
        EvidenceClassification.HISTORICAL
        if historical
        else EvidenceClassification.CURRENT
    )
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description="Connection acquisition delays were observed.",
        observed_at=timestamp,
        service="payment-api",
        classification=classification,
    )


def _finding(evidence_id="logs:one"):
    return RCAFinding(
        finding_id="finding:one",
        summary="Current evidence supports a probable connection delay cause.",
        root_cause="Connection acquisition delay",
        supporting_evidence_ids=[evidence_id],
        confidence=0.7,
        confidence_explanation="The cited source supports this probable cause.",
        causal_reasoning="Requests waited while connections were acquired.",
    )


def _report(evidence=None, *, timeline=None, recommendation_result=None):
    evidence = evidence or _evidence()
    finding = _finding(evidence.evidence_id)
    if recommendation_result is None:
        recommendation_items = [
            {
                "recommendation_id": f"recommendation:{category.value}",
                "title": f"Review {category.value} connection evidence",
                "description": "Review the cited connection evidence.",
                "category": category,
                "rationale": "The RCA cites connection acquisition delays.",
                "supporting_evidence_ids": [evidence.evidence_id],
                "related_finding_ids": [finding.finding_id],
                "verification_steps": ["Compare the cited records."],
                "confidence": 0.6,
            }
            for category in RecommendationCategory
        ]
        recommendation_result = RecommendationResult(
            status=RecommendationStatus.COMPLETED,
            incident_id="INC-SCHEMA",
            recommendations=recommendation_items,
            corrective_action_count=1,
            preventive_action_count=1,
            diagnostic_action_count=1,
            overall_summary="A diagnostic recommendation is available.",
            source_rca_finding_ids=[finding.finding_id],
        )
    if timeline is None:
        timeline = [
            RCATimelineEvent(
                event_id="event:one",
                timestamp=evidence.observed_at,
                service=evidence.service,
                source=evidence.source_type,
                description="Connection acquisition delay was recorded.",
                evidence_ids=[evidence.evidence_id],
                event_type="degradation",
            )
        ]
    report = InvestigationReport(
        incident_id="INC-SCHEMA",
        incident_summary="Elevated payment API latency.",
        affected_service="payment-api",
        investigation_status=InvestigationStatus.SUCCESS,
        report_status=ReportStatus.COMPLETED,
        executive_summary=ReportExecutiveSummary(
            summary=GroundedReportText(
                text="Current evidence supports a probable connection delay.",
                evidence_ids=[evidence.evidence_id],
                finding_ids=[finding.finding_id],
            ),
            investigation_outcome=GroundedReportText(
                text="A diagnostic action is available.",
                evidence_ids=[evidence.evidence_id],
                finding_ids=[finding.finding_id],
                recommendation_ids=[
                    f"recommendation:{category.value}"
                    for category in RecommendationCategory
                ],
            ),
            root_cause_summary=finding.summary,
            root_cause_finding_id=finding.finding_id,
            key_recommendation_ids=[
                f"recommendation:{category.value}"
                for category in RecommendationCategory
            ],
        ),
        incident_overview=IncidentOverview(
            incident_id="INC-SCHEMA",
            title="Elevated payment API latency",
            description="Elevated payment API latency.",
            affected_service="payment-api",
            affected_services=["payment-api"],
        ),
        evidence_references=[
            ReportEvidenceReference(
                **evidence.model_dump(mode="python"),
                linked_finding_ids=[finding.finding_id],
                linked_recommendation_ids=[
                    f"recommendation:{category.value}"
                    for category in RecommendationCategory
                ],
            )
        ],
        evidence_analysis=EvidenceAnalysis(
            historical_rag_evidence_ids=(
                [evidence.evidence_id]
                if evidence.classification is EvidenceClassification.HISTORICAL
                else []
            ),
            application_log_evidence_ids=(
                [evidence.evidence_id]
                if evidence.source_type is EvidenceSourceType.APPLICATION_LOGS
                else []
            ),
            verified_facts=[
                RCAEvidenceClaim(
                    description="Connection delay was observed.",
                    evidence_ids=[evidence.evidence_id],
                    classification=evidence.classification,
                )
            ],
        ),
        timeline=timeline,
        root_cause_analysis=RootCauseAnalysisSection(
            status=RCAStatus.ANALYZED,
            primary_finding=finding,
            confidence=finding.confidence,
            confidence_explanation=finding.confidence_explanation,
            analysis_summary="A probable cause was identified.",
        ),
        recommendation_result=recommendation_result,
        recommendation_sections=RecommendationSections(
            corrective_action_ids=["recommendation:corrective"],
            preventive_action_ids=["recommendation:preventive"],
            diagnostic_action_ids=["recommendation:diagnostic"],
        ),
        conclusion=InvestigationConclusion(
            summary=GroundedReportText(
                text="The finding should be verified.",
                evidence_ids=[evidence.evidence_id],
                finding_ids=[finding.finding_id],
                recommendation_ids=[
                    f"recommendation:{category.value}"
                    for category in RecommendationCategory
                ],
            ),
            methodology=["Only validated RCA findings are reported."],
        ),
    )
    return report


def test_report_schema_preserves_evidence_and_finding_references():
    evidence = _evidence()
    report = _report(evidence)

    assert report.evidence_references[0].evidence_id == evidence.evidence_id
    assert report.root_cause_analysis.primary_finding.finding_id == "finding:one"
    assert report.recommendation_sections.diagnostic_action_ids == [
        "recommendation:diagnostic"
    ]


def test_report_rejects_duplicate_evidence_references():
    report = _report()
    payload = report.model_dump(mode="python")
    payload["evidence_references"].append(payload["evidence_references"][0])

    with pytest.raises(ValidationError, match="must be unique"):
        InvestigationReport.model_validate(payload)


def test_report_rejects_unknown_evidence_and_finding_ids():
    report = _report()
    payload = report.model_dump(mode="python")
    payload["executive_summary"]["summary"]["evidence_ids"] = ["unknown:evidence"]

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        InvestigationReport.model_validate(payload)

    payload = report.model_dump(mode="python")
    payload["executive_summary"]["summary"]["finding_ids"] = ["unknown:finding"]
    with pytest.raises(ValidationError, match="unknown finding IDs"):
        InvestigationReport.model_validate(payload)

    payload = report.model_dump(mode="python")
    payload["conclusion"]["summary"]["recommendation_ids"] = [
        "recommendation:missing"
    ]
    with pytest.raises(ValidationError, match="unknown recommendation IDs"):
        InvestigationReport.model_validate(payload)


def test_report_rejects_recommendation_section_ids_not_in_result():
    report = _report()
    payload = report.model_dump(mode="python")
    payload["recommendation_sections"]["diagnostic_action_ids"] = [
        "recommendation:missing"
    ]

    with pytest.raises(ValidationError, match="account for every recommendation"):
        InvestigationReport.model_validate(payload)


def test_report_rejects_non_chronological_timeline():
    later = _evidence("logs:later", timestamp=_TIME + timedelta(minutes=1))
    earlier = _evidence("logs:earlier", timestamp=_TIME)
    first = RCATimelineEvent(
        event_id="event:later",
        timestamp=later.observed_at,
        source=later.source_type,
        description=later.description,
        evidence_ids=[later.evidence_id],
        event_type="degradation",
    )
    second = RCATimelineEvent(
        event_id="event:earlier",
        timestamp=earlier.observed_at,
        source=earlier.source_type,
        description=earlier.description,
        evidence_ids=[earlier.evidence_id],
        event_type="symptom",
    )
    report = _report(later)
    payload = report.model_dump(mode="python")
    payload["evidence_references"].append(
        ReportEvidenceReference(**earlier.model_dump(mode="python"))
    )
    payload["evidence_analysis"]["application_log_evidence_ids"].append(
        earlier.evidence_id
    )
    payload["timeline"] = [first.model_dump(mode="python"), second.model_dump(mode="python")]

    with pytest.raises(ValidationError, match="chronological order"):
        InvestigationReport.model_validate(payload)


def test_report_rejects_historical_evidence_as_primary_support():
    report = _report()
    payload = report.model_dump(mode="python")
    evidence = payload["evidence_references"][0]
    evidence.update(
        evidence_id="history:only",
        source_type=EvidenceSourceType.HISTORICAL_RAG,
        source_name=EvidenceSourceType.HISTORICAL_RAG.value,
        classification=EvidenceClassification.HISTORICAL,
    )
    payload["evidence_analysis"]["application_log_evidence_ids"] = []
    payload["evidence_analysis"]["historical_rag_evidence_ids"] = ["history:only"]
    payload["evidence_analysis"]["verified_facts"][0].update(
        evidence_ids=["history:only"],
        classification=EvidenceClassification.HISTORICAL,
    )
    payload["timeline"][0].update(
        evidence_ids=["history:only"],
        source=EvidenceSourceType.HISTORICAL_RAG,
    )
    payload["root_cause_analysis"]["primary_finding"]["supporting_evidence_ids"] = [
        "history:only"
    ]
    for recommendation in payload["recommendation_result"]["recommendations"]:
        recommendation["supporting_evidence_ids"] = ["history:only"]
    payload["executive_summary"]["summary"]["evidence_ids"] = ["history:only"]
    payload["executive_summary"]["investigation_outcome"]["evidence_ids"] = [
        "history:only"
    ]
    payload["conclusion"]["summary"]["evidence_ids"] = ["history:only"]

    with pytest.raises(ValidationError, match="current operational evidence"):
        InvestigationReport.model_validate(payload)
