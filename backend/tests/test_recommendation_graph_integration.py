"""LangGraph integration tests for recommendation routing and state isolation."""

import asyncio

import pytest
from app.graph import workflow
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
)
from app.schemas.rca import RCAFinding, RCAResult, RCAStatus
from app.schemas.recommendations import RecommendationStatus
from app.schemas.supervisor import (
    DecisionStatus,
    EvidenceSufficiency,
    InvestigationDecision,
    InvestigationEvidenceSummary,
    RecommendedNextStage,
    SourceAssessment,
    SourceAssessmentStatus,
)

_CURRENT_EVIDENCE_ID = "logs:current"
_CAUSE = "Database connection pool saturation"


def _request(incident_id: str = "INC-REC-GRAPH") -> IncidentRequest:
    return IncidentRequest(
        incident_id=incident_id,
        description="Database connection errors with elevated latency.",
        affected_service="payment-api",
    )


def _rca_result(evidence_id: str, cause: str = _CAUSE) -> RCAResult:
    return RCAResult(
        status=RCAStatus.ANALYZED,
        incident_summary="Payment API database connection errors.",
        primary_finding=RCAFinding(
            finding_id="finding:primary",
            summary=cause,
            root_cause=cause,
            affected_services=["payment-api"],
            supporting_evidence_ids=[evidence_id],
            confidence=0.8,
            confidence_explanation="Current evidence supports this cause.",
            causal_reasoning=f"Evidence supports {cause}.",
        ),
        overall_confidence=0.8,
        confidence_explanation="Current evidence supports the RCA.",
        analysis_summary="The primary cause was validated.",
    )


def _recommendation_payload(evidence_id: str, finding_id: str, cause: str) -> dict:
    shared = {
        "related_finding_ids": [finding_id],
        "supporting_evidence_ids": [evidence_id],
        "affected_services": ["payment-api"],
        "verification_steps": [f"Verify the current evidence for {cause}."],
        "confidence": 0.8,
    }
    return {
        "recommendations": [
            {
                **shared,
                "recommendation_id": f"rec:{evidence_id}:corrective",
                "title": f"Mitigate {cause}",
                "description": f"Mitigate the validated {cause} and verify recovery.",
                "category": "corrective",
                "rationale": f"This addresses {cause} supported by current evidence.",
                "requires_human_approval": False,
            },
            {
                **shared,
                "recommendation_id": f"rec:{evidence_id}:preventive",
                "title": f"Monitor recurrence of {cause}",
                "description": f"Monitor recurrence of the validated {cause}.",
                "category": "preventive",
                "rationale": f"Recurrence monitoring is relevant to {cause}.",
                "requires_human_approval": False,
            },
            {
                **shared,
                "recommendation_id": f"rec:{evidence_id}:diagnostic",
                "title": f"Verify {cause}",
                "description": f"Read-only inspect the evidence for {cause}.",
                "category": "diagnostic",
                "rationale": f"Verification can resolve uncertainty about {cause}.",
                "requires_human_approval": False,
            },
        ]
    }


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, _prompt):
        return self.response


def _ready_decision(evidence_id: str) -> InvestigationDecision:
    return InvestigationDecision(
        incident_id="INC-REC-GRAPH",
        decision_status=DecisionStatus.READY_FOR_RCA,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        evidence_summary=InvestigationEvidenceSummary(),
        source_assessments=[
            SourceAssessment(
                source_type=EvidenceSourceType.APPLICATION_LOGS,
                classification=EvidenceClassification.CURRENT,
                status=SourceAssessmentStatus.AVAILABLE,
                evidence_count=1,
                evidence_ids=[evidence_id],
                summary="Current application log evidence is available.",
            )
        ],
        supporting_evidence_ids=[evidence_id],
        historical_current_alignment="no_historical",
        confidence=0.8,
        decision_rationale="The current evidence is ready for RCA.",
        recommended_next_stage=RecommendedNextStage.ROOT_CAUSE_ANALYSIS,
    )


def _install_graph_fakes(monkeypatch, *, cause: str = _CAUSE):
    async def log_node(state):
        evidence_id = f"{state['incident_id']}:logs"
        item = Evidence(
            evidence_id=evidence_id,
            source_type=EvidenceSourceType.APPLICATION_LOGS,
            source_name="application logs",
            description=f"Current evidence supports {cause}.",
            service="payment-api",
            classification=EvidenceClassification.CURRENT,
        )
        return {"log_evidence": [item]}

    async def empty_branch(_state):
        return {}

    monkeypatch.setattr(workflow.nodes, "log_analysis_node", log_node)
    for name in (
        "metrics_analysis_node",
        "prometheus_analysis_node",
        "grafana_analysis_node",
    ):
        monkeypatch.setattr(workflow.nodes, name, empty_branch)
    monkeypatch.setattr(workflow.nodes, "historical_rag_node", lambda _state: {})

    async def supervisor(state, *, llm=None):
        del llm
        evidence_id = f"{state['incident_id']}:logs"
        decision = _ready_decision(evidence_id).model_copy(
            update={"incident_id": state["incident_id"]}
        )
        return {"supervisor_decision": decision}

    async def rca(state, *, llm=None):
        del llm
        evidence_id = f"{state['incident_id']}:logs"
        return {"rca_result": _rca_result(evidence_id, cause)}

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", rca)
    monkeypatch.setattr(
        workflow,
        "_investigation_graph",
        workflow.build_investigation_graph(),
    )


def test_valid_rca_routes_to_recommendation_and_persists_typed_result(monkeypatch):
    _install_graph_fakes(monkeypatch)
    evidence_id = "INC-REC-GRAPH:logs"
    model = FakeStructuredModel(
        _recommendation_payload(evidence_id, "finding:primary", _CAUSE)
    )

    state = asyncio.run(
        workflow.run_investigation(
            _request(),
            recommendation_llm=model,
        )
    )

    assert model.schema is not None
    assert state["rca_result"].status is RCAStatus.ANALYZED
    assert state["recommendation_result"].status is RecommendationStatus.COMPLETED
    assert state["recommendation_result"].incident_id == "INC-REC-GRAPH"
    assert len(state["recommendation_result"].recommendations) == 3
    assert state["recommendations"] == []
    assert [item.evidence_id for item in state["consolidated_evidence"]] == [
        evidence_id
    ]


@pytest.mark.parametrize(
    "status",
    [
        RCAStatus.INSUFFICIENT_EVIDENCE,
        RCAStatus.PARTIAL_SUCCESS,
        RCAStatus.SKIPPED,
        RCAStatus.FAILED,
    ],
)
def test_non_actionable_rca_terminates_without_recommendation_node(
    monkeypatch,
    status,
):
    _install_graph_fakes(monkeypatch)

    async def non_actionable_rca(_state, *, llm=None):
        del llm
        return {
            "rca_result": RCAResult(
                status=status,
                incident_summary="No validated cause.",
                overall_confidence=0,
                confidence_explanation="The RCA did not establish a cause.",
                analysis_summary="Recommendation generation must not continue.",
            )
        }

    async def fail_recommendation(*_args, **_kwargs):
        pytest.fail("The recommendation node must not run for a non-actionable RCA.")

    monkeypatch.setattr(workflow.nodes, "recommendation_node", fail_recommendation)
    monkeypatch.setattr(workflow.nodes, "root_cause_analysis_node", non_actionable_rca)
    monkeypatch.setattr(
        workflow,
        "_investigation_graph",
        workflow.build_investigation_graph(),
    )
    state = asyncio.run(workflow.run_investigation(_request()))

    assert state["recommendation_result"] is None


def test_route_rejects_missing_invalid_and_non_analyzed_rca():
    assert (
        workflow._route_after_rca({"rca_result": None}) == "report_generation"
    )
    assert (
        workflow._route_after_rca({"rca_result": {"status": "unknown"}})
        == "report_generation"
    )
    assert (
        workflow._route_after_rca({"rca_result": _rca_result("logs:missing")})
        == "report_generation"
    )
    assert (
        workflow._route_after_rca(
            {
                "rca_result": RCAResult(
                    status=RCAStatus.INSUFFICIENT_EVIDENCE,
                    incident_summary="No cause.",
                    overall_confidence=0,
                    confidence_explanation="None.",
                    analysis_summary="Insufficient.",
                )
            }
        )
        == "report_generation"
    )


def test_repeated_graph_runs_do_not_share_recommendation_state(monkeypatch):
    _install_graph_fakes(monkeypatch)
    first_evidence = "INC-FIRST:logs"
    second_evidence = "INC-SECOND:logs"

    first = asyncio.run(
        workflow.run_investigation(
            _request("INC-FIRST"),
            recommendation_llm=FakeStructuredModel(
                _recommendation_payload(first_evidence, "finding:primary", _CAUSE)
            ),
        )
    )
    second = asyncio.run(
        workflow.run_investigation(
            _request("INC-SECOND"),
            recommendation_llm=FakeStructuredModel(
                _recommendation_payload(second_evidence, "finding:primary", _CAUSE)
            ),
        )
    )

    assert first["recommendation_result"].incident_id == "INC-FIRST"
    assert second["recommendation_result"].incident_id == "INC-SECOND"
    assert first["recommendation_result"].recommendations[
        0
    ].supporting_evidence_ids == [first_evidence]
    assert second["recommendation_result"].recommendations[
        0
    ].supporting_evidence_ids == [second_evidence]
    assert first["recommendation_result"] is not second["recommendation_result"]
