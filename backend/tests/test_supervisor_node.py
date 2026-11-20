"""Tests for Supervisor integration with shared investigation state."""

import asyncio
from unittest.mock import AsyncMock

from app.agents.supervisor_agent import assess_investigation
from app.graph import nodes
from app.graph.state import create_initial_state
from app.schemas import DecisionStatus, IncidentRequest


def _state():
    return create_initial_state(
        IncidentRequest(
            incident_id="INC-SUPERVISOR-NODE",
            description="Elevated payment latency",
            affected_service="payment-api",
        )
    )


def test_supervisor_node_sets_decision_without_replacing_evidence_or_findings():
    state = _state()
    state["errors"] = ["existing branch error"]
    before = state.copy()

    update = asyncio.run(nodes.supervisor_node(state))

    assert update["supervisor_decision"].decision_status is DecisionStatus.NO_EVIDENCE
    assert update["supervisor_decision"].evidence_summary.hypotheses == []
    assert state == before
    assert set(update) == {
        "supervisor_decision",
        "execution_metadata",
        "warnings",
        "errors",
    }
    assert update["errors"] == []
    metadata = next(iter(update["execution_metadata"].values()))
    assert metadata["agent_name"] == "supervisor"
    assert metadata["decision_status"] == DecisionStatus.NO_EVIDENCE.value


def test_supervisor_node_forwards_injected_model(monkeypatch):
    state = _state()
    result = asyncio.run(assess_investigation(state))
    llm = object()
    assessor = AsyncMock(return_value=result)
    monkeypatch.setattr(nodes, "assess_investigation", assessor)

    update = asyncio.run(nodes.supervisor_node(state, llm=llm))

    assessor.assert_awaited_once_with(state, llm=llm)
    assert update["supervisor_decision"] is result.decision


def test_supervisor_node_records_failure_and_uses_deterministic_fallback(monkeypatch):
    state = _state()
    fallback = asyncio.run(assess_investigation(state))
    assessor = AsyncMock(
        side_effect=[RuntimeError("private model response"), fallback]
    )
    monkeypatch.setattr(nodes, "assess_investigation", assessor)

    update = asyncio.run(nodes.supervisor_node(state, llm=object()))

    assert assessor.await_count == 2
    assert assessor.await_args_list[1].kwargs == {"llm": None}
    assert update["supervisor_decision"] is fallback.decision
    metadata = next(iter(update["execution_metadata"].values()))
    assert metadata["status"] == "failed"
    assert metadata["fallback_used"] is True
    assert metadata["error_type"] == "RuntimeError"
    assert update["errors"] == ["supervisor failed (RuntimeError)."]
    assert any("deterministic fallback" in item for item in update["warnings"])
    assert "private model response" not in repr(update)
