"""Graph routing, report persistence, and report-state isolation tests."""

import asyncio
import json
from datetime import UTC, datetime

from app.graph import workflow
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
    RCAEventType,
    RCAFinding,
    RCAResult,
    RCAStatus,
    RCATimelineEvent,
    RecommendationCategory,
    RecommendationResult,
    RecommendationStatus,
    ReportNarrative,
    ReportStatus,
)
from app.schemas.investigation_report import GroundedReportText
from app.schemas.rca import RCAEvidenceClaim
from app.schemas.supervisor import (
    DecisionStatus,
    EvidenceSufficiency,
    HistoricalCurrentAlignment,
    InvestigationDecision,
    InvestigationEvidenceSummary,
    RecommendedNextStage,
    SourceAssessment,
    SourceAssessmentStatus,
)

_NOW = datetime(2026, 12, 1, 10, 0, tzinfo=UTC)
_BRANCH_FIELDS = {
    "historical_rag": (
        "historical_evidence",
        EvidenceSourceType.HISTORICAL_RAG,
        EvidenceClassification.HISTORICAL,
    ),
    "log_analysis": (
        "log_evidence",
        EvidenceSourceType.APPLICATION_LOGS,
        EvidenceClassification.CURRENT,
    ),
    "metrics_analysis": (
        "metrics_evidence",
        EvidenceSourceType.SERVICE_METRICS,
        EvidenceClassification.CURRENT,
    ),
    "prometheus_analysis": (
        "prometheus_evidence",
        EvidenceSourceType.PROMETHEUS,
        EvidenceClassification.CURRENT,
    ),
    "grafana_analysis": (
        "grafana_evidence",
        EvidenceSourceType.GRAFANA,
        EvidenceClassification.CURRENT,
    ),
}


def _request(incident_id):
    return IncidentRequest(
        incident_id=incident_id,
        description="Elevated payment API latency.",
        affected_service="payment-api",
    )


def _install_branches(monkeypatch):
    for branch_name, (field, source_type, classification) in _BRANCH_FIELDS.items():
        async def handler(
            state,
            *,
            target=field,
            source=source_type,
            kind=classification,
            name=branch_name,
        ):
            evidence = Evidence(
                evidence_id=f"{state['incident_id']}:{name}",
                source_type=source,
                source_name=source.value,
                description=f"{source.value} evidence for payment API latency.",
                observed_at=_NOW,
                service="payment-api",
                classification=kind,
            )
            return {target: [evidence]}

        node_function = {
            "historical_rag": "historical_rag_node",
            "log_analysis": "log_analysis_node",
            "metrics_analysis": "metrics_analysis_node",
            "prometheus_analysis": "prometheus_analysis_node",
            "grafana_analysis": "grafana_analysis_node",
        }[branch_name]
        monkeypatch.setattr(workflow.nodes, node_function, handler)


def _decision(state, status=DecisionStatus.READY_FOR_RCA):
    assessments = [
        SourceAssessment(
            source_type=evidence.source_type,
            classification=evidence.classification,
            status=SourceAssessmentStatus.AVAILABLE,
            evidence_count=1,
            evidence_ids=[evidence.evidence_id],
            summary=f"{evidence.source_name} supplied evidence.",
        )
        for evidence in state["consolidated_evidence"]
    ]
    current_ids = [
        evidence.evidence_id
        for evidence in state["consolidated_evidence"]
        if evidence.classification is EvidenceClassification.CURRENT
    ]
    if status is DecisionStatus.READY_FOR_RCA:
        sufficiency = EvidenceSufficiency.SUFFICIENT
        next_stage = RecommendedNextStage.ROOT_CAUSE_ANALYSIS
    else:
        sufficiency = EvidenceSufficiency.PARTIAL
        next_stage = RecommendedNextStage.ADDITIONAL_EVIDENCE_COLLECTION
    return InvestigationDecision(
        incident_id=state["incident_id"],
        decision_status=status,
        evidence_sufficiency=sufficiency,
        evidence_summary=InvestigationEvidenceSummary(),
        source_assessments=assessments,
        supporting_evidence_ids=current_ids[:1],
        historical_current_alignment=HistoricalCurrentAlignment.NOT_COMPARABLE,
        confidence=0.8,
        decision_rationale="The decision is based on the collected evidence.",
        recommended_next_stage=next_stage,
    )


def _rca(state):
    log = next(
        item
        for item in state["consolidated_evidence"]
        if item.source_type is EvidenceSourceType.APPLICATION_LOGS
    )
    finding_id = f"{state['incident_id']}:finding"
    return RCAResult(
        status=RCAStatus.ANALYZED,
        incident_summary=state["incident_description"],
        primary_finding=RCAFinding(
            finding_id=finding_id,
            summary="Current logs support a connection acquisition delay.",
            root_cause="Connection acquisition delay",
            affected_services=["payment-api"],
            supporting_evidence_ids=[log.evidence_id],
            confidence=0.7,
            confidence_explanation="The cited current log supports this cause.",
            causal_reasoning="Requests waited during connection acquisition.",
        ),
        timeline=[
            RCATimelineEvent(
                event_id=f"{state['incident_id']}:event",
                timestamp=log.observed_at,
                service=log.service,
                source=log.source_type,
                description=log.description,
                evidence_ids=[log.evidence_id],
                event_type=RCAEventType.DEGRADATION,
            )
        ],
        verified_facts=[
            RCAEvidenceClaim(
                description="The application logs report connection delays.",
                evidence_ids=[log.evidence_id],
                classification=EvidenceClassification.CURRENT,
            )
        ],
        overall_confidence=0.7,
        confidence_explanation="The evidence supports a probable cause.",
        analysis_summary="A current log supports the finding.",
    )


def _recommendations(state):
    finding_id = f"{state['incident_id']}:finding"
    log_id = f"{state['incident_id']}:log_analysis"
    items = []
    for category in RecommendationCategory:
        suffix = category.value
        items.append(
            {
                "recommendation_id": f"{state['incident_id']}:{suffix}",
                "title": f"Review {suffix} evidence",
                "description": "Review the connection acquisition evidence.",
                "category": category,
                "rationale": "The validated RCA identifies connection delays.",
                "supporting_evidence_ids": [log_id],
                "related_finding_ids": [finding_id],
                "verification_steps": ["Compare the cited log evidence."],
                "confidence": 0.7,
            }
        )
    return RecommendationResult(
        status=RecommendationStatus.COMPLETED,
        incident_id=state["incident_id"],
        recommendations=items,
        corrective_action_count=1,
        preventive_action_count=1,
        diagnostic_action_count=1,
        overall_summary="All recommendation categories are linked to the RCA.",
        source_rca_finding_ids=[finding_id],
    )


class ContextualReportModel:
    def __init__(self):
        self.schema = None
        self.invocations = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        self.invocations += 1
        content = prompt.to_messages()[-1].content
        context = json.loads(content.split("\n", maxsplit=1)[1])
        evidence = context["current_evidence"]
        evidence_ids = [
            item["evidence_id"]
            for item in evidence
            if item["source_type"] == EvidenceSourceType.APPLICATION_LOGS.value
        ]
        rca = context["validated_rca"]
        finding = rca["primary_finding"] if rca else None
        finding_ids = [finding["finding_id"]] if finding else []
        recommendations = context["validated_recommendations"]
        recommendation_ids = (
            [
                item["recommendation_id"]
                for item in recommendations["recommendations"]
            ]
            if recommendations
            else []
        )
        return ReportNarrative(
            executive_summary=GroundedReportText(
                text=(
                    finding["summary"]
                    if finding
                    else "No validated root-cause conclusion is available."
                ),
                evidence_ids=evidence_ids,
                finding_ids=finding_ids,
            ),
            investigation_outcome=GroundedReportText(
                text=(
                    "All recommendation categories are linked to the RCA."
                    if recommendation_ids
                    else "No validated RCA result was available; no cause is asserted."
                ),
                evidence_ids=evidence_ids,
                finding_ids=finding_ids,
                recommendation_ids=recommendation_ids,
            ),
            conclusion=GroundedReportText(
                text=(
                    finding["root_cause"]
                    if finding
                    else "No root cause was established."
                ),
                evidence_ids=evidence_ids,
                finding_ids=finding_ids,
                recommendation_ids=recommendation_ids,
            ),
        )


def test_report_runs_after_recommendations_and_is_isolated_per_investigation(
    monkeypatch,
):
    _install_branches(monkeypatch)
    calls = []

    async def supervisor(state, *, llm=None):
        calls.append((state["incident_id"], "supervisor"))
        return {"supervisor_decision": _decision(state)}

    async def root_cause(state, *, llm=None):
        calls.append((state["incident_id"], "rca"))
        return {"rca_result": _rca(state)}

    async def recommendations(state, *, llm=None):
        calls.append((state["incident_id"], "recommendations"))
        return {"recommendation_result": _recommendations(state)}

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", root_cause)
    monkeypatch.setattr(workflow.nodes, "recommendation_node", recommendations)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )
    model = ContextualReportModel()

    async def run_both():
        return await asyncio.gather(
            workflow.run_investigation(_request("INC-REPORT-A"), report_llm=model),
            workflow.run_investigation(_request("INC-REPORT-B"), report_llm=model),
        )

    first, second = asyncio.run(run_both())

    assert model.schema is ReportNarrative
    assert model.invocations == 2
    assert first["investigation_report"].report_status is ReportStatus.COMPLETED
    assert first["final_report"] is first["investigation_report"]
    assert first["investigation_report"].recommendation_result == first[
        "recommendation_result"
    ]
    assert second["investigation_report"].incident_id == "INC-REPORT-B"
    first_ids = {
        item.evidence_id
        for item in first["investigation_report"].evidence_references
    }
    second_ids = {
        item.evidence_id
        for item in second["investigation_report"].evidence_references
    }
    assert first_ids.isdisjoint(second_ids)
    assert all(item.startswith("INC-REPORT-A:") for item in first_ids)
    assert all(item.startswith("INC-REPORT-B:") for item in second_ids)
    assert [name for incident, name in calls if incident == "INC-REPORT-A"] == [
        "supervisor",
        "rca",
        "recommendations",
    ]


def test_non_ready_supervisor_routes_directly_to_partial_report(monkeypatch):
    _install_branches(monkeypatch)
    calls = []

    async def supervisor(state, *, llm=None):
        return {
            "supervisor_decision": _decision(
                state,
                status=DecisionStatus.NEEDS_MORE_EVIDENCE,
            )
        }

    async def unexpected_rca(*_args, **_kwargs):
        calls.append("rca")
        raise AssertionError("RCA must not run for a non-ready decision.")

    async def unexpected_recommendations(*_args, **_kwargs):
        calls.append("recommendations")
        raise AssertionError("Recommendations must not run without an analyzed RCA.")

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", unexpected_rca)
    monkeypatch.setattr(
        workflow.nodes,
        "recommendation_node",
        unexpected_recommendations,
    )
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )

    result = asyncio.run(
        workflow.run_investigation(
            _request("INC-REPORT-PARTIAL"),
            report_llm=ContextualReportModel(),
        )
    )

    assert calls == []
    assert result["rca_result"] is None
    assert result["recommendation_result"] is None
    assert result["investigation_report"].report_status is ReportStatus.INSUFFICIENT_EVIDENCE
    assert result["investigation_report"].root_cause_analysis.primary_finding is None
    assert "Root cause analysis" in result["investigation_report"].missing_sections
