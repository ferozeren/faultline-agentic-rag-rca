"""Validation tests for the foundational investigation schemas."""

from datetime import UTC, datetime

import pytest
from app.schemas import (
    AgentObservation,
    Evidence,
    EvidenceClassification,
    EvidenceGap,
    EvidenceSourceType,
    IncidentRequest,
    InvestigationReport,
    InvestigationStatus,
    Recommendation,
    RootCauseFinding,
)
from pydantic import ValidationError


def _evidence(source_type: EvidenceSourceType) -> Evidence:
    return Evidence(
        evidence_id=f"{source_type.value}-1",
        source_type=source_type,
        source_name=source_type.value,
        description="Observed connection pool saturation.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"count": 12, "attributes": {"region": "west"}},
    )


def test_valid_incident_request():
    request = IncidentRequest(
        incident_id="INC-42",
        description="  Elevated request latency  ",
        affected_service="checkout",
        start_time="2026-01-01T10:00:00Z",
        end_time="2026-01-01T10:30:00Z",
        additional_context={"region": "west", "attempts": 2},
    )

    assert request.description == "Elevated request latency"
    assert request.start_time < request.end_time
    assert request.additional_context == {"region": "west", "attempts": 2}


@pytest.mark.parametrize("description", ["", " ", "   "])
def test_empty_incident_description_is_rejected(description):
    with pytest.raises(ValidationError, match="description"):
        IncidentRequest(description=description)


def test_invalid_incident_time_range_is_rejected():
    with pytest.raises(ValidationError, match="later than start"):
        IncidentRequest(
            description="Elevated latency",
            start_time="2026-01-01T11:00:00Z",
            end_time="2026-01-01T10:00:00Z",
        )


def test_uncomparable_incident_timezones_are_rejected():
    with pytest.raises(ValidationError, match="timezone-aware or naive"):
        IncidentRequest(
            description="Elevated latency",
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_valid_evidence_creation():
    evidence = _evidence(EvidenceSourceType.APPLICATION_LOGS)

    assert evidence.evidence_id == "application_logs-1"
    assert evidence.metadata["attributes"] == {"region": "west"}
    assert evidence.classification is EvidenceClassification.CURRENT


def test_historical_evidence_is_classified_separately():
    evidence = _evidence(EvidenceSourceType.HISTORICAL_RAG)

    assert evidence.classification is EvidenceClassification.HISTORICAL


@pytest.mark.parametrize(
    "source_type",
    [
        EvidenceSourceType.APPLICATION_LOGS,
        EvidenceSourceType.SERVICE_METRICS,
        EvidenceSourceType.PROMETHEUS,
        EvidenceSourceType.GRAFANA,
    ],
)
def test_operational_evidence_is_current(source_type):
    assert _evidence(source_type).classification is EvidenceClassification.CURRENT


def test_evidence_rejects_misclassified_source():
    with pytest.raises(ValidationError, match="must be classified"):
        Evidence(
            evidence_id="historical-1",
            source_type=EvidenceSourceType.HISTORICAL_RAG,
            source_name="incident archive",
            description="Prior incident observation.",
            classification=EvidenceClassification.CURRENT,
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_root_cause_confidence_boundaries_are_valid(confidence):
    finding = RootCauseFinding(
        finding_id="cause-1",
        probable_cause="Connection pool saturation",
        confidence=confidence,
        explanation="The logs and metrics show exhausted connections.",
    )

    assert finding.confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_root_cause_confidence_out_of_range_is_rejected(confidence):
    with pytest.raises(ValidationError, match="confidence"):
        RootCauseFinding(
            finding_id="cause-1",
            probable_cause="Connection pool saturation",
            confidence=confidence,
            explanation="Evidence is not conclusive.",
        )


def test_root_cause_confidence_rejects_booleans():
    with pytest.raises(ValidationError, match="confidence"):
        RootCauseFinding(
            finding_id="cause-1",
            probable_cause="Connection pool saturation",
            confidence=True,
            explanation="Evidence is not conclusive.",
        )


def test_recommendation_is_advisory_and_structured():
    recommendation = Recommendation(
        recommendation_id="recommendation-1",
        action="Review the connection pool limits.",
        rationale="Current evidence shows saturation.",
        priority="high",
        expected_outcome="Confirm whether capacity is constrained.",
        supporting_evidence_ids=["metrics-1"],
        risk_or_caution="Review expected load before changing limits.",
    )

    assert recommendation.priority.value == "high"
    assert recommendation.advisory_only is True
    with pytest.raises(ValidationError, match="advisory_only"):
        Recommendation(
            recommendation_id="recommendation-1",
            action="Review the connection pool limits.",
            rationale="Current evidence shows saturation.",
            priority="high",
            expected_outcome="Confirm whether capacity is constrained.",
            advisory_only=False,
        )


def test_investigation_report_creation_uses_nested_models():
    historical = AgentObservation(
        observation_id="observation-h",
        agent_name="historical_rag",
        summary="A prior incident involved connection saturation.",
        evidence_ids=["historical_rag-1"],
        source_types=[EvidenceSourceType.HISTORICAL_RAG],
        classification=EvidenceClassification.HISTORICAL,
    )
    current = AgentObservation(
        observation_id="observation-c",
        agent_name="log_analysis",
        summary="Current logs show slow database requests.",
        evidence_ids=["application_logs-1"],
        source_types=[EvidenceSourceType.APPLICATION_LOGS],
        classification=EvidenceClassification.CURRENT,
    )
    report = InvestigationReport(
        incident_id="INC-42",
        incident_summary="Elevated request latency",
        affected_service="checkout",
        investigation_status=InvestigationStatus.PARTIAL,
        observations=[historical, current],
        historical_findings=[historical],
        current_findings=[current],
        evidence_gaps=[
            EvidenceGap(
                gap_id="gap-1",
                description="No application trace data was available.",
            )
        ],
        limitations=["Only a subset of operational sources was available."],
    )

    assert (
        report.historical_findings[0].classification
        is EvidenceClassification.HISTORICAL
    )
    assert report.current_findings[0].classification is EvidenceClassification.CURRENT
    assert report.generated_at.tzinfo is not None


def test_report_rejects_historical_observation_in_current_findings():
    observation = AgentObservation(
        observation_id="observation-h",
        agent_name="historical_rag",
        summary="Prior event.",
        classification=EvidenceClassification.HISTORICAL,
    )

    with pytest.raises(ValidationError, match="current_findings"):
        InvestigationReport(
            incident_summary="Current incident",
            investigation_status=InvestigationStatus.COMPLETED,
            current_findings=[observation],
        )
