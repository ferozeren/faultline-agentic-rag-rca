"""Tests for Supervisor-gated RCA routing in the investigation graph."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from app.graph import nodes, workflow
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
    RCAResult,
)
from app.schemas.rca import RCAStatus
from app.schemas.supervisor import (
    ConflictType,
    DecisionStatus,
    EvidenceConflict,
    EvidenceSufficiency,
    InvestigationDecision,
    InvestigationEvidenceSummary,
    RecommendedNextStage,
    SourceAssessment,
    SourceAssessmentStatus,
)

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


def _request(incident_id="INC-RCA-GRAPH"):
    return IncidentRequest(
        incident_id=incident_id,
        description="Elevated payment latency.",
        affected_service="payment-api",
    )


def _evidence(source_type, classification, evidence_id):
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description=f"{source_type.value} reports a payment service anomaly.",
        observed_at=datetime(2026, 11, 6, 10, 0, tzinfo=UTC),
        service="payment-api",
        classification=classification,
    )


def _decision(status):
    assessments = []
    conflicts = []
    if status is DecisionStatus.READY_FOR_RCA:
        next_stage = RecommendedNextStage.ROOT_CAUSE_ANALYSIS
        sufficiency = EvidenceSufficiency.SUFFICIENT
    elif status is DecisionStatus.NEEDS_MORE_EVIDENCE:
        next_stage = RecommendedNextStage.ADDITIONAL_EVIDENCE_COLLECTION
        sufficiency = EvidenceSufficiency.PARTIAL
    elif status is DecisionStatus.CONFLICTING_EVIDENCE:
        next_stage = RecommendedNextStage.EVIDENCE_CONFLICT_REVIEW
        sufficiency = EvidenceSufficiency.PARTIAL
        assessments = [
            SourceAssessment(
                source_type=EvidenceSourceType.APPLICATION_LOGS,
                classification=EvidenceClassification.CURRENT,
                status=SourceAssessmentStatus.AVAILABLE,
                evidence_count=2,
                evidence_ids=["logs:a", "logs:b"],
                summary="Two current records conflict.",
            )
        ]
        conflicts = [
            EvidenceConflict(
                conflict_type=ConflictType.CURRENT_CURRENT,
                evidence_ids=["logs:a", "logs:b"],
                description="Current observations conflict.",
            )
        ]
    else:
        next_stage = RecommendedNextStage.MANUAL_REVIEW
        sufficiency = EvidenceSufficiency.INSUFFICIENT
    return InvestigationDecision(
        incident_id="INC-RCA-GRAPH",
        decision_status=status,
        evidence_sufficiency=sufficiency,
        evidence_summary=InvestigationEvidenceSummary(),
        source_assessments=assessments,
        conflicts=conflicts,
        supporting_evidence_ids=[],
        historical_current_alignment="no_historical",
        confidence=0.8,
        decision_rationale="Decision for graph routing test.",
        recommended_next_stage=next_stage,
    )


def _install_branches(monkeypatch, *, incident_scoped=False):
    for branch_name, (field, source_type, classification) in _BRANCH_FIELDS.items():

        async def handler(
            state,
            *,
            name=branch_name,
            target=field,
            source=source_type,
            kind=classification,
        ):
            incident_prefix = f"{state['incident_id']}:" if incident_scoped else ""
            item = _evidence(source, kind, f"{incident_prefix}{name}:evidence")
            return {
                target: [item],
                "execution_metadata": {
                    f"{name}:rca-test": {
                        "agent_name": name,
                        "status": "success",
                    }
                },
            }

        node_function = {
            "historical_rag": "historical_rag_node",
            "log_analysis": "log_analysis_node",
            "metrics_analysis": "metrics_analysis_node",
            "prometheus_analysis": "prometheus_analysis_node",
            "grafana_analysis": "grafana_analysis_node",
        }[branch_name]
        monkeypatch.setattr(workflow.nodes, node_function, handler)


def _result(incident_summary="RCA result"):
    return RCAResult(
        status=RCAStatus.INSUFFICIENT_EVIDENCE,
        incident_summary=incident_summary,
        overall_confidence=0,
        confidence_explanation="No model analysis in this graph test.",
        analysis_summary="Test result with no causal claim.",
    )


def test_ready_decision_runs_rca_after_aggregation_and_persists_result(monkeypatch):
    _install_branches(monkeypatch)
    supervisor_calls = []
    rca_calls = []
    decision = _decision(DecisionStatus.READY_FOR_RCA)

    async def supervisor(state, *, llm=None):
        supervisor_calls.append(state)
        return {
            "supervisor_decision": decision,
            "execution_metadata": {
                "supervisor:test": {"agent_name": "supervisor", "status": "success"}
            },
            "errors": [],
            "warnings": [],
        }

    async def rca(state, *, llm=None):
        rca_calls.append((state, llm))
        return {"rca_result": _result(state["incident_id"])}

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", rca)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )
    model = object()

    result = asyncio.run(workflow.run_investigation(_request(), rca_llm=model))

    assert len(supervisor_calls) == 1
    assert len(supervisor_calls[0]["consolidated_evidence"]) == 5
    assert len(rca_calls) == 1
    assert len(rca_calls[0][0]["consolidated_evidence"]) == 5
    assert rca_calls[0][1] is model
    assert result["supervisor_decision"] is decision
    assert result["rca_result"].incident_summary == "INC-RCA-GRAPH"
    assert len(result["consolidated_evidence"]) == 5


@pytest.mark.parametrize(
    "status",
    [
        DecisionStatus.NEEDS_MORE_EVIDENCE,
        DecisionStatus.CONFLICTING_EVIDENCE,
        DecisionStatus.NO_EVIDENCE,
    ],
)
def test_non_ready_supervisor_status_ends_before_rca(monkeypatch, status):
    _install_branches(monkeypatch)
    decision = _decision(status)
    rca = AsyncMock(return_value={"rca_result": _result()})
    monkeypatch.setattr(
        workflow.nodes,
        "supervisor_node",
        AsyncMock(
            return_value={
                "supervisor_decision": decision,
                "execution_metadata": {},
                "errors": [],
                "warnings": [],
            }
        ),
    )
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", rca)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )

    result = asyncio.run(workflow.run_investigation(_request()))

    rca.assert_not_awaited()
    assert result["rca_result"] is None
    assert result["supervisor_decision"] is decision


def test_missing_supervisor_decision_ends_before_rca(monkeypatch):
    _install_branches(monkeypatch)
    rca = AsyncMock(return_value={"rca_result": _result()})
    monkeypatch.setattr(
        workflow.nodes,
        "supervisor_node",
        AsyncMock(return_value={"execution_metadata": {}}),
    )
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", rca)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )

    result = asyncio.run(workflow.run_investigation(_request()))

    rca.assert_not_awaited()
    assert result["supervisor_decision"] is None
    assert result["rca_result"] is None


def test_conditional_router_accepts_only_ready_status():
    assert (
        workflow._route_after_supervisor(
            {"supervisor_decision": _decision(DecisionStatus.READY_FOR_RCA)}
        )
        == "root_cause_analysis"
    )
    for status in (
        DecisionStatus.NEEDS_MORE_EVIDENCE,
        DecisionStatus.CONFLICTING_EVIDENCE,
        DecisionStatus.NO_EVIDENCE,
    ):
        assert (
            workflow._route_after_supervisor({"supervisor_decision": _decision(status)})
            == "report_generation"
        )
    assert (
        workflow._route_after_supervisor({"supervisor_decision": None})
        == "report_generation"
    )
    assert (
        workflow._route_after_supervisor(
            {"supervisor_decision": {"decision_status": "ready_for_rca"}}
        )
        == "report_generation"
    )


def test_rca_node_refuses_non_ready_and_missing_decisions(monkeypatch):
    agent = AsyncMock()
    monkeypatch.setattr(nodes, "analyze_root_cause", agent)

    skipped = asyncio.run(
        nodes.root_cause_analysis_node(
            {"incident_description": "Test", "supervisor_decision": None}
        )
    )

    agent.assert_not_awaited()
    assert skipped["rca_result"].status is RCAStatus.SKIPPED
    assert skipped["warnings"]


def test_rca_failure_preserves_evidence_and_prior_investigation_results(monkeypatch):
    _install_branches(monkeypatch)
    decision = _decision(DecisionStatus.READY_FOR_RCA)

    async def supervisor(state, *, llm=None):
        return {"supervisor_decision": decision, "execution_metadata": {}}

    async def fail(state, *, llm=None):
        raise RuntimeError("private model response")

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(nodes, "analyze_root_cause", fail)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(result["consolidated_evidence"]) == 5
    assert result["supervisor_decision"] is decision
    assert result["rca_result"].status is RCAStatus.FAILED
    assert result["errors"] == ["RCA analysis failed (RuntimeError)."]
    assert "private model response" not in repr(result)


def test_repeated_rca_investigations_do_not_leak_between_runs(monkeypatch):
    _install_branches(monkeypatch, incident_scoped=True)
    decision = _decision(DecisionStatus.READY_FOR_RCA)
    seen = []

    async def supervisor(state, *, llm=None):
        return {"supervisor_decision": decision, "execution_metadata": {}}

    async def rca(state, *, llm=None):
        seen.append([item.evidence_id for item in state["consolidated_evidence"]])
        return {"rca_result": _result(state["incident_id"])}

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", rca)
    monkeypatch.setattr(
        workflow, "_investigation_graph", workflow.build_investigation_graph()
    )

    first, second = asyncio.run(
        _run_twice(_request("INC-FIRST-RCA"), _request("INC-SECOND-RCA"))
    )

    assert all(value.startswith("INC-FIRST-RCA:") for value in seen[0])
    assert all(value.startswith("INC-SECOND-RCA:") for value in seen[1])
    assert first["rca_result"].incident_summary == "INC-FIRST-RCA"
    assert second["rca_result"].incident_summary == "INC-SECOND-RCA"
    assert first["consolidated_evidence"] is not second["consolidated_evidence"]


async def _run_twice(first, second):
    return (
        await workflow.run_investigation(first),
        await workflow.run_investigation(second),
    )
