"""Tests for the executable parallel investigation workflow."""

import asyncio
import copy

import pytest
from app.graph import workflow
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
)
from app.schemas.investigation_report import InvestigationStatus
from pydantic import ValidationError

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


def _request(**updates):
    values = {
        "incident_id": "INC-WORKFLOW",
        "description": "Elevated payment latency",
        "affected_service": "payment-api",
        "start_time": "2026-11-06T10:00:00Z",
        "end_time": "2026-11-06T10:30:00Z",
        "additional_context": {"region": "west", "nested": {"ticket": "OPS-21"}},
    }
    values.update(updates)
    return IncidentRequest.model_validate(values)


def _evidence(
    source_type: EvidenceSourceType,
    classification: EvidenceClassification,
    evidence_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id or f"{source_type.value}:one",
        source_type=source_type,
        source_name=source_type.value,
        description=f"Evidence from {source_type.value}.",
        classification=classification,
    )


def _branch_update(
    branch_name: str,
    *,
    evidence: list[Evidence] | None = None,
    status: str = "success",
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
):
    evidence_field, _, _ = _BRANCH_FIELDS[branch_name]
    return {
        evidence_field: evidence or [],
        "execution_metadata": {
            f"{branch_name}:test-run": {
                "agent_name": branch_name,
                "status": status,
                "evidence_count": len(evidence or []),
            }
        },
        "errors": errors or [],
        "warnings": warnings or [],
    }


def _edge_pairs(compiled_graph):
    return {(edge.source, edge.target) for edge in compiled_graph.get_graph().edges}


def _install_branch_mocks(monkeypatch, handlers):
    for branch_name in _BRANCH_FIELDS:
        function_name = {
            "historical_rag": "historical_rag_node",
            "log_analysis": "log_analysis_node",
            "metrics_analysis": "metrics_analysis_node",
            "prometheus_analysis": "prometheus_analysis_node",
            "grafana_analysis": "grafana_analysis_node",
        }[branch_name]
        monkeypatch.setattr(
            workflow.nodes,
            function_name,
            handlers[branch_name],
        )


def _successful_branch_handlers():
    handlers = {}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        evidence = _evidence(source_type, classification)

        async def handler(state, *, name=branch_name, item=evidence):
            return _branch_update(name, evidence=[item])

        handlers[branch_name] = handler
    return handlers


def test_graph_registers_nodes_and_fans_out_to_a_multi_source_join():
    graph = workflow.build_investigation_graph()
    graph_view = graph.get_graph()
    node_names = set(graph_view.nodes)
    edges = _edge_pairs(graph)

    assert (
        set(workflow.SPECIALIST_NODE_NAMES)
        | {
            "evidence_aggregation",
            "supervisor",
            "root_cause_analysis",
            "recommendation",
            "report_generation",
        }
    ) <= node_names
    assert {("__start__", name) for name in workflow.SPECIALIST_NODE_NAMES} <= edges
    assert {
        (name, "evidence_aggregation") for name in workflow.SPECIALIST_NODE_NAMES
    } <= edges
    assert ("evidence_aggregation", "supervisor") in edges
    assert ("supervisor", "report_generation") in edges
    assert ("supervisor", "root_cause_analysis") in edges
    assert ("root_cause_analysis", "report_generation") in edges
    assert ("root_cause_analysis", "recommendation") in edges
    assert ("recommendation", "report_generation") in edges
    assert ("report_generation", "__end__") in edges
    assert not any(
        (left, right) in edges
        for left in workflow.SPECIALIST_NODE_NAMES
        for right in workflow.SPECIALIST_NODE_NAMES
    )


def test_parallel_branches_receive_the_same_initial_incident_and_reach_aggregation(
    monkeypatch,
):
    calls = {}
    branches = {}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        evidence = _evidence(source_type, classification)

        async def handler(state, *, name=branch_name, item=evidence):
            calls[name] = copy.deepcopy(state)
            return _branch_update(name, evidence=[item])

        branches[branch_name] = handler
    _install_branch_mocks(monkeypatch, branches)

    state = asyncio.run(workflow.run_investigation(_request()))

    assert set(calls) == set(_BRANCH_FIELDS)
    for branch_state in calls.values():
        assert branch_state["incident_id"] == "INC-WORKFLOW"
        assert branch_state["incident_description"] == "Elevated payment latency"
        assert branch_state["affected_service"] == "payment-api"
        assert branch_state["start_time"] == state["start_time"]
        assert branch_state["end_time"] == state["end_time"]
        assert branch_state["log_evidence"] == []
        assert branch_state["historical_evidence"] == []
    assert len(state["consolidated_evidence"]) == 5
    assert {item.source_type for item in state["consolidated_evidence"]} == {
        source_type for _, source_type, _ in _BRANCH_FIELDS.values()
    }
    assert {item.classification for item in state["consolidated_evidence"]} == {
        EvidenceClassification.HISTORICAL,
        EvidenceClassification.CURRENT,
    }
    assert state["investigation_status"] is InvestigationStatus.SUCCESS


def test_all_five_branches_execute_concurrently(monkeypatch):
    all_started = asyncio.Event()
    entered = set()
    branches = {}

    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        evidence = _evidence(source_type, classification)

        async def handler(state, *, name=branch_name, item=evidence):
            entered.add(name)
            if len(entered) == len(_BRANCH_FIELDS):
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=5)
            return _branch_update(name, evidence=[item])

        branches[branch_name] = handler
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(
        asyncio.wait_for(workflow.run_investigation(_request()), timeout=10)
    )

    assert entered == set(_BRANCH_FIELDS)
    assert len(result["consolidated_evidence"]) == 5


def test_aggregation_waits_for_each_branch_completion(monkeypatch):
    completed = set()
    aggregation_calls = []
    branches = {}
    original_aggregation = workflow.nodes.evidence_aggregation_node

    for branch_name in _BRANCH_FIELDS:

        async def handler(state, *, name=branch_name):
            completed.add(name)
            return _branch_update(name)

        branches[branch_name] = handler
    _install_branch_mocks(monkeypatch, branches)

    def observe_aggregation(state):
        aggregation_calls.append(set(completed))
        return original_aggregation(state)

    monkeypatch.setattr(
        workflow.nodes,
        "evidence_aggregation_node",
        observe_aggregation,
    )
    monkeypatch.setattr(
        workflow,
        "_investigation_graph",
        workflow.build_investigation_graph(),
    )
    asyncio.run(workflow.run_investigation(_request()))

    assert aggregation_calls == [set(_BRANCH_FIELDS)]


def test_supervisor_runs_once_after_aggregation_and_preserves_evidence(monkeypatch):
    _install_branch_mocks(monkeypatch, _successful_branch_handlers())
    original_aggregation = workflow.nodes.evidence_aggregation_node
    aggregation_done = []
    supervisor_calls = []

    def observe_aggregation(state):
        result = original_aggregation(state)
        aggregation_done.append(result["consolidated_evidence"])
        return result

    async def observe_supervisor(state, *, llm=None):
        supervisor_calls.append((state, llm))
        return {"supervisor_decision": None}

    monkeypatch.setattr(
        workflow.nodes,
        "evidence_aggregation_node",
        observe_aggregation,
    )
    monkeypatch.setattr(workflow.nodes, "supervisor_node", observe_supervisor)
    monkeypatch.setattr(
        workflow,
        "_investigation_graph",
        workflow.build_investigation_graph(),
    )

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(aggregation_done) == 1
    assert len(supervisor_calls) == 1
    assert len(supervisor_calls[0][0]["consolidated_evidence"]) == 5
    assert len(result["consolidated_evidence"]) == 5


def test_one_failed_branch_keeps_other_evidence_and_reports_partial_status(
    monkeypatch,
):
    branches = {}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        if branch_name == "prometheus_analysis":

            async def fail(state):
                raise RuntimeError("private endpoint details")

            branches[branch_name] = fail
            continue
        evidence = _evidence(source_type, classification)

        async def succeed(state, *, name=branch_name, item=evidence):
            return _branch_update(name, evidence=[item])

        branches[branch_name] = succeed
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(result["consolidated_evidence"]) == 4
    assert result["prometheus_evidence"] == []
    assert result["investigation_status"] is InvestigationStatus.PARTIAL_SUCCESS
    assert result["errors"] == ["prometheus_analysis failed (RuntimeError)."]
    assert any(
        "prometheus_analysis" in gap.description for gap in result["evidence_gaps"]
    )
    assert "private endpoint details" not in repr(result)


def test_multiple_failed_branches_preserve_successful_evidence(monkeypatch):
    branches = {}
    failing = {"metrics_analysis", "grafana_analysis"}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        if branch_name in failing:

            async def fail(state, *, name=branch_name):
                raise ValueError(f"{name} private details")

            branches[branch_name] = fail
            continue
        evidence = _evidence(source_type, classification)

        async def succeed(state, *, name=branch_name, item=evidence):
            return _branch_update(name, evidence=[item])

        branches[branch_name] = succeed
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(result["consolidated_evidence"]) == 3
    assert result["investigation_status"] is InvestigationStatus.PARTIAL_SUCCESS
    assert len(result["errors"]) == 2
    assert {
        branch
        for branch in failing
        if any(branch in error for error in result["errors"])
    } == failing


def test_all_operational_branches_can_fail_while_historical_evidence_survives(
    monkeypatch,
):
    historical = _evidence(
        EvidenceSourceType.HISTORICAL_RAG,
        EvidenceClassification.HISTORICAL,
    )
    branches = {
        "historical_rag": lambda state: _branch_update(
            "historical_rag",
            evidence=[historical],
        )
    }
    for branch_name in (
        "log_analysis",
        "metrics_analysis",
        "prometheus_analysis",
        "grafana_analysis",
    ):

        async def fail(state, *, name=branch_name):
            raise RuntimeError(f"{name} private details")

        branches[branch_name] = fail
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert result["historical_evidence"] == [historical]
    assert result["consolidated_evidence"] == [historical]
    assert result["investigation_status"] is InvestigationStatus.PARTIAL_SUCCESS
    assert len(result["errors"]) == 4
    failed_statuses = {
        metadata["agent_name"]: metadata["status"]
        for metadata in result["execution_metadata"].values()
        if metadata.get("agent_name")
        in {
            "log_analysis",
            "metrics_analysis",
            "prometheus_analysis",
            "grafana_analysis",
        }
    }
    assert failed_statuses == {
        "log_analysis": "failed",
        "metrics_analysis": "failed",
        "prometheus_analysis": "failed",
        "grafana_analysis": "failed",
    }


def test_all_failed_branches_return_failed_status_and_gaps(monkeypatch):
    async def fail(state):
        raise RuntimeError("private tool response")

    _install_branch_mocks(
        monkeypatch,
        {branch_name: fail for branch_name in _BRANCH_FIELDS},
    )

    result = asyncio.run(workflow.run_investigation(_request()))

    assert result["consolidated_evidence"] == []
    assert result["investigation_status"] is InvestigationStatus.FAILED
    assert len(result["errors"]) == 5
    assert len(result["evidence_gaps"]) == 5


def test_all_branches_without_evidence_return_no_evidence_without_conclusions(
    monkeypatch,
):
    async def empty(state, *, name):
        return _branch_update(name, status="no_evidence")

    handlers = {
        branch_name: (lambda state, name=branch_name: empty(state, name=name))
        for branch_name in _BRANCH_FIELDS
    }
    _install_branch_mocks(monkeypatch, handlers)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert result["consolidated_evidence"] == []
    assert result["investigation_status"] is InvestigationStatus.NO_EVIDENCE
    assert result["selected_root_cause"] is None
    assert result["root_cause_findings"] == []


def test_branch_errors_warnings_and_execution_metadata_are_preserved(monkeypatch):
    branches = {}
    for branch_name in _BRANCH_FIELDS:

        async def return_partial(state, *, name=branch_name):
            return _branch_update(
                name,
                status="partial_success",
                errors=[f"{name} reported a tool error."],
                warnings=[f"{name} returned incomplete data."],
            )

        branches[branch_name] = return_partial
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(result["execution_metadata"]) == 7
    assert any(
        metadata.get("agent_name") == "supervisor"
        for metadata in result["execution_metadata"].values()
    )
    assert len(result["errors"]) == 5
    assert len(result["warnings"]) == 5
    assert result["investigation_status"] is InvestigationStatus.PARTIAL_SUCCESS


def test_duplicate_evidence_uses_existing_idempotent_reducers(monkeypatch):
    duplicated = _evidence(
        EvidenceSourceType.APPLICATION_LOGS,
        EvidenceClassification.CURRENT,
        "log:duplicate",
    )
    branches = {}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():
        evidence = (
            [duplicated, duplicated]
            if branch_name == "log_analysis"
            else [_evidence(source_type, classification)]
        )

        async def handler(state, *, name=branch_name, items=evidence):
            return _branch_update(name, evidence=items)

        branches[branch_name] = handler
    _install_branch_mocks(monkeypatch, branches)

    result = asyncio.run(workflow.run_investigation(_request()))

    assert len(result["log_evidence"]) == 1
    assert len(result["consolidated_evidence"]) == 5


def test_repeated_invocations_are_isolated_and_do_not_mutate_request(monkeypatch):
    branches = {}
    for branch_name, (_, source_type, classification) in _BRANCH_FIELDS.items():

        async def handler(
            state, *, name=branch_name, source=source_type, kind=classification
        ):
            return _branch_update(
                name,
                evidence=[_evidence(source, kind, f"{state['incident_id']}:{name}")],
            )

        branches[branch_name] = handler
    _install_branch_mocks(monkeypatch, branches)
    first = _request(incident_id="INC-FIRST")
    second = _request(incident_id="INC-SECOND")
    first_before = first.model_dump(mode="python")
    second_before = second.model_dump(mode="python")

    first_result, second_result = asyncio.run(_run_twice(first, second))

    assert first.model_dump(mode="python") == first_before
    assert second.model_dump(mode="python") == second_before
    assert first_result["incident_id"] == "INC-FIRST"
    assert second_result["incident_id"] == "INC-SECOND"
    assert all(
        item.evidence_id.startswith("INC-FIRST:")
        for item in first_result["consolidated_evidence"]
    )
    assert all(
        item.evidence_id.startswith("INC-SECOND:")
        for item in second_result["consolidated_evidence"]
    )
    assert first_result["additional_context"] is not first.additional_context


def test_optional_llm_is_forwarded_to_historical_rag_only(monkeypatch):
    seen_llms = []

    def historical(state, *, llm=None):
        seen_llms.append(llm)
        return _branch_update("historical_rag", status="no_evidence")

    branches = {"historical_rag": historical}
    for branch_name in (
        "log_analysis",
        "metrics_analysis",
        "prometheus_analysis",
        "grafana_analysis",
    ):

        async def empty(state, *, name=branch_name):
            return _branch_update(name, status="no_evidence")

        branches[branch_name] = empty
    _install_branch_mocks(monkeypatch, branches)
    llm = object()

    asyncio.run(workflow.run_investigation(_request(), llm=llm))

    assert seen_llms == [llm]


def test_optional_supervisor_llm_is_forwarded_to_supervisor_only(monkeypatch):
    seen_llms = []

    async def supervisor(state, *, llm=None):
        seen_llms.append(llm)
        return {"supervisor_decision": None}

    monkeypatch.setattr(workflow.nodes, "supervisor_node", supervisor)
    monkeypatch.setattr(
        workflow,
        "_investigation_graph",
        workflow.build_investigation_graph(),
    )
    branches = {}
    for branch_name in _BRANCH_FIELDS:

        async def empty(state, *, name=branch_name):
            return _branch_update(name, status="no_evidence")

        branches[branch_name] = empty
    _install_branch_mocks(monkeypatch, branches)
    llm = object()

    asyncio.run(workflow.run_investigation(_request(), supervisor_llm=llm))

    assert seen_llms == [llm]


async def _run_twice(first, second):
    return (
        await workflow.run_investigation(first),
        await workflow.run_investigation(second),
    )


def test_invalid_incident_input_is_rejected():
    with pytest.raises(ValidationError):
        asyncio.run(workflow.run_investigation({"description": "   "}))


def test_public_workflow_api_and_mermaid_diagram_are_available():
    assert workflow._investigation_graph is not None
    diagram = workflow.get_investigation_graph_diagram()

    assert "START" in diagram
    assert "historical_rag" in diagram
    assert "log_analysis" in diagram
    assert "metrics_analysis" in diagram
    assert "prometheus_analysis" in diagram
    assert "grafana_analysis" in diagram
    assert "evidence_aggregation" in diagram
    assert "supervisor" in diagram
    assert "root_cause_analysis" in diagram
    assert "END" in diagram
