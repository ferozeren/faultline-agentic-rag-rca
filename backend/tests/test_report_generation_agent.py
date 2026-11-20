"""Tests for source-grounded report generation and safe model fallback."""

import asyncio
import copy
from datetime import UTC, datetime, timedelta

import pytest
from app.agents.report_generation_agent import generate_investigation_report
from app.graph import create_initial_state
from app.schemas import (
    AgentObservation,
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
    RCAEventType,
    RCAEvidenceClaim,
    RCAFinding,
    RCAHypothesis,
    RCAHypothesisStatus,
    RCAResult,
    RCAStatus,
    RCATimelineEvent,
    RecommendationCategory,
    RecommendationResult,
    RecommendationStatus,
)
from app.schemas.investigation_report import (
    GroundedReportText,
    ReportNarrative,
    ReportStatus,
)
from app.schemas.supervisor import ConflictType, EvidenceConflict

_NOW = datetime(2026, 12, 1, 10, 0, tzinfo=UTC)


def _evidence(
    evidence_id: str,
    source_type: EvidenceSourceType,
    description: str,
    *,
    observed_at: datetime = _NOW,
) -> Evidence:
    classification = (
        EvidenceClassification.HISTORICAL
        if source_type is EvidenceSourceType.HISTORICAL_RAG
        else EvidenceClassification.CURRENT
    )
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description=description,
        observed_at=observed_at,
        service="payment-api",
        classification=classification,
    )


def _rca() -> RCAResult:
    return RCAResult(
        status=RCAStatus.ANALYZED,
        incident_summary="Elevated payment API latency.",
        primary_finding=RCAFinding(
            finding_id="finding:db-pool",
            summary="Current evidence supports database connection pool saturation.",
            root_cause="Database connection pool saturation",
            affected_services=["payment-api"],
            supporting_evidence_ids=["logs:db", "metrics:db"],
            contradicting_evidence_ids=[],
            confidence=0.82,
            confidence_explanation="Current log and service metric observations align.",
            causal_reasoning="Requests waited for database connections.",
            contributing_factors=[
                RCAEvidenceClaim(
                    description="Connection acquisition delays are present.",
                    evidence_ids=["logs:db"],
                    classification=EvidenceClassification.CURRENT,
                )
            ],
        ),
        alternative_hypotheses=[
            RCAHypothesis(
                hypothesis_id="hypothesis:downstream",
                description="A downstream dependency may contribute to latency.",
                supporting_evidence_ids=["metrics:db"],
                contradicting_evidence_ids=["historical:db"],
                confidence=0.3,
                status=RCAHypothesisStatus.UNRESOLVED,
            )
        ],
        timeline=[
            RCATimelineEvent(
                event_id="event:pool-wait",
                timestamp=_NOW,
                service="payment-api",
                source=EvidenceSourceType.APPLICATION_LOGS,
                description="Database connection acquisition delay was logged.",
                evidence_ids=["logs:db"],
                event_type=RCAEventType.DEGRADATION,
            )
        ],
        verified_facts=[
            RCAEvidenceClaim(
                description="The logs include connection acquisition delays.",
                evidence_ids=["logs:db"],
                classification=EvidenceClassification.CURRENT,
            )
        ],
        hypotheses=[],
        unresolved_conflicts=[
            EvidenceConflict(
                conflict_type=ConflictType.CURRENT_CURRENT,
                evidence_ids=["logs:db", "metrics:db"],
                description="Log and metric observations differ in their timing.",
            )
        ],
        overall_confidence=0.82,
        confidence_explanation="Current evidence supports a probable cause.",
        analysis_summary="Current evidence supports the connection-pool finding.",
    )


def _recommendations() -> RecommendationResult:
    actions = []
    for category, suffix in (
        (RecommendationCategory.CORRECTIVE, "corrective"),
        (RecommendationCategory.PREVENTIVE, "preventive"),
        (RecommendationCategory.DIAGNOSTIC, "diagnostic"),
    ):
        actions.append(
            {
                "recommendation_id": f"recommendation:{suffix}",
                "title": f"Review {suffix} connection-pool evidence",
                "description": "Review the connection acquisition observations.",
                "category": category,
                "priority": "medium",
                "urgency": "planned",
                "rationale": "The validated RCA identifies connection-pool saturation.",
                "supporting_evidence_ids": ["logs:db"],
                "related_finding_ids": ["finding:db-pool"],
                "affected_services": ["payment-api"],
                "verification_steps": ["Compare the cited log and metric evidence."],
                "confidence": 0.7,
            }
        )
    return RecommendationResult(
        status=RecommendationStatus.COMPLETED,
        incident_id="INC-REPORT",
        recommendations=actions,
        corrective_action_count=1,
        preventive_action_count=1,
        diagnostic_action_count=1,
        overall_summary="Three actions are linked to the validated RCA.",
        source_rca_finding_ids=["finding:db-pool"],
    )


def _state():
    request = IncidentRequest(
        incident_id="INC-REPORT",
        description="Elevated payment API latency.",
        affected_service="payment-api",
        environment="production",
        severity="high",
        start_time=_NOW - timedelta(minutes=10),
        end_time=_NOW,
    )
    state = create_initial_state(request)
    historical = _evidence(
        "historical:db",
        EvidenceSourceType.HISTORICAL_RAG,
        "A prior incident discussed database connection saturation.",
    )
    logs = _evidence(
        "logs:db",
        EvidenceSourceType.APPLICATION_LOGS,
        "Current database connection acquisition delays are logged.",
    )
    metrics = _evidence(
        "metrics:db",
        EvidenceSourceType.SERVICE_METRICS,
        "Current service metric reports elevated connection waits.",
        observed_at=_NOW + timedelta(minutes=1),
    )
    state["historical_evidence"] = [historical]
    state["log_evidence"] = [logs]
    state["metrics_evidence"] = [metrics]
    state["consolidated_evidence"] = [historical, logs, metrics]
    state["observations"] = [
        AgentObservation(
            observation_id="observation:logs",
            agent_name="log_analysis",
            summary="Connection acquisition delays are logged.",
            evidence_ids=["logs:db"],
            source_types=[EvidenceSourceType.APPLICATION_LOGS],
            classification=EvidenceClassification.CURRENT,
        )
    ]
    state["rag_observations"] = [
        AgentObservation(
            observation_id="observation:history",
            agent_name="historical_rag",
            summary="A prior incident discussed a similar connection issue.",
            evidence_ids=["historical:db"],
            source_types=[EvidenceSourceType.HISTORICAL_RAG],
            classification=EvidenceClassification.HISTORICAL,
        )
    ]
    state["investigation_status"] = "success"
    state["rca_result"] = _rca()
    state["recommendation_result"] = _recommendations()
    return state


def _narrative(
    *,
    evidence_ids=None,
    finding_ids=None,
    recommendation_ids=None,
) -> ReportNarrative:
    return ReportNarrative(
        executive_summary=GroundedReportText(
            text="Current evidence supports database connection pool saturation.",
            evidence_ids=evidence_ids or ["logs:db", "metrics:db"],
            finding_ids=finding_ids or ["finding:db-pool"],
        ),
        investigation_outcome=GroundedReportText(
            text="Log and metric observations differ in their timing.",
            evidence_ids=evidence_ids or ["logs:db", "metrics:db"],
            finding_ids=finding_ids or ["finding:db-pool"],
            recommendation_ids=recommendation_ids
            or [
                "recommendation:corrective",
                "recommendation:preventive",
                "recommendation:diagnostic",
            ],
        ),
        conclusion=GroundedReportText(
            text="Database connection pool saturation",
            evidence_ids=evidence_ids or ["logs:db", "metrics:db"],
            finding_ids=finding_ids or ["finding:db-pool"],
            recommendation_ids=recommendation_ids
            or [
                "recommendation:corrective",
                "recommendation:preventive",
                "recommendation:diagnostic",
            ],
        ),
    )


class FakeStructuredModel:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else _narrative()
        self.error = error
        self.schema = None
        self.prompt = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.response


class TimeoutModel(FakeStructuredModel):
    async def ainvoke(self, _prompt):
        await asyncio.sleep(10)


def test_report_preserves_source_data_and_separates_historical_context():
    state = _state()
    original = copy.deepcopy(state)
    model = FakeStructuredModel()

    report = asyncio.run(generate_investigation_report(state, llm=model))

    assert model.schema is ReportNarrative
    assert report.report_status is ReportStatus.COMPLETED
    assert report.root_cause_analysis.primary_finding.finding_id == "finding:db-pool"
    assert report.recommendation_result == state["recommendation_result"]
    assert report.evidence_analysis.historical_rag_evidence_ids == ["historical:db"]
    assert set(report.evidence_analysis.application_log_evidence_ids) == {"logs:db"}
    assert report.evidence_analysis.service_metric_evidence_ids == ["metrics:db"]
    assert report.timeline[0].timestamp == _NOW
    assert report.evidence_references[0].evidence_id == "historical:db"
    historical = next(
        item
        for item in report.evidence_references
        if item.evidence_id == "historical:db"
    )
    assert historical.linked_finding_ids == []
    assert report.executive_summary.business_impact is None
    assert report.unresolved_issues
    assert state == original


def test_report_model_receives_safe_log_summary_without_raw_metadata():
    state = _state()
    logs = state["log_evidence"][0].model_copy(
        update={
            "description": (
                "Log MCP operational record; error_code=GATEWAY_TIMEOUT; "
                "http_status=504"
            ),
            "metadata": {
                "user_id": "private-user",
                "source_ip": "192.0.2.1",
                "error_code": "GATEWAY_TIMEOUT",
            },
            "raw_reference": "private-request-id",
        }
    )
    state["log_evidence"] = [logs]
    state["consolidated_evidence"] = [
        state["historical_evidence"][0],
        logs,
        state["metrics_evidence"][0],
    ]
    model = FakeStructuredModel()

    report = asyncio.run(generate_investigation_report(state, llm=model))

    prompt = model.prompt.to_string()
    assert report.report_status is ReportStatus.COMPLETED
    assert "GATEWAY_TIMEOUT" in prompt
    assert "private-user" not in prompt
    assert "192.0.2.1" not in prompt
    assert "private-request-id" not in prompt


def test_invalid_rca_references_produce_insufficient_report_without_cause():
    state = _state()
    invalid_rca = state["rca_result"].model_dump(mode="python")
    invalid_rca["primary_finding"]["supporting_evidence_ids"] = ["unknown:evidence"]
    state["rca_result"] = invalid_rca
    state["recommendation_result"] = None

    report = asyncio.run(generate_investigation_report(state))

    assert report.report_status is ReportStatus.INSUFFICIENT_EVIDENCE
    assert report.root_cause is None
    assert report.root_cause_analysis.primary_finding is None
    assert "Root cause analysis" in report.missing_sections
    assert any("rejected the RCA" in error for error in report.errors)


def test_invalid_recommendation_references_are_not_copied_into_report():
    state = _state()
    invalid_result = state["recommendation_result"].model_dump(mode="python")
    invalid_result["recommendations"][0]["supporting_evidence_ids"] = [
        "unknown:evidence"
    ]
    state["recommendation_result"] = invalid_result

    report = asyncio.run(generate_investigation_report(state))

    assert report.report_status is ReportStatus.PARTIAL
    assert report.root_cause_analysis.primary_finding.finding_id == "finding:db-pool"
    assert report.recommendation_result is None
    assert report.recommendation_sections.corrective_action_ids == []
    assert any("rejected recommendations" in error for error in report.errors)


def test_historical_only_evidence_never_becomes_a_current_root_cause():
    state = _state()
    historical = state["historical_evidence"][0]
    state["log_evidence"] = []
    state["metrics_evidence"] = []
    state["consolidated_evidence"] = [historical]
    state["rca_result"] = None
    state["recommendation_result"] = None

    report = asyncio.run(generate_investigation_report(state))

    assert report.report_status is ReportStatus.INSUFFICIENT_EVIDENCE
    assert report.root_cause_analysis.primary_finding is None
    assert report.evidence_analysis.historical_rag_evidence_ids == ["historical:db"]
    assert report.evidence_analysis.application_log_evidence_ids == []
    assert "Current operational evidence" in report.missing_sections


def test_model_output_with_unknown_ids_falls_back_and_records_error():
    state = _state()
    invalid = _narrative(
        evidence_ids=["invented:evidence"],
        finding_ids=["invented:finding"],
        recommendation_ids=["invented:recommendation"],
    )
    model = FakeStructuredModel(response=invalid)

    report = asyncio.run(generate_investigation_report(state, llm=model))

    assert report.report_status is ReportStatus.PARTIAL
    assert report.root_cause_analysis.primary_finding.finding_id == "finding:db-pool"
    assert report.executive_summary.summary.evidence_ids == [
        "logs:db",
        "metrics:db",
    ]
    assert any("narrative generation failed" in error for error in report.errors)
    assert report.warnings


@pytest.mark.parametrize(
    ("model", "error_name"),
    [
        (FakeStructuredModel(error=RuntimeError("model unavailable")), "RuntimeError"),
        (TimeoutModel(), "TimeoutError"),
    ],
)
def test_model_failure_and_timeout_return_grounded_partial_reports(model, error_name):
    report = asyncio.run(
        generate_investigation_report(_state(), llm=model, timeout_seconds=0.01)
    )

    assert report.report_status is ReportStatus.PARTIAL
    assert report.root_cause_analysis.primary_finding.finding_id == "finding:db-pool"
    assert any(error_name in error for error in report.errors)
    assert report.recommendation_result.recommendations
