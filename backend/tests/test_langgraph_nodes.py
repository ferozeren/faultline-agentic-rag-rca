"""Tests for independently callable specialist and evidence nodes."""

import asyncio
import copy

from app.graph import create_initial_state, nodes
from app.graph.state import merge_evidence, merge_observations
from app.schemas import (
    AgentObservation,
    Evidence,
    EvidenceClassification,
    EvidenceGap,
    EvidenceSourceType,
    IncidentRequest,
)


def _state():
    return create_initial_state(
        IncidentRequest(
            description="Elevated request latency",
            affected_service="checkout",
        )
    )


def _evidence(
    evidence_id: str,
    source_type: EvidenceSourceType,
    classification: EvidenceClassification,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description=f"Evidence {evidence_id}.",
        classification=classification,
    )


def _observation(observation_id: str, summary: str) -> AgentObservation:
    return AgentObservation(
        observation_id=observation_id,
        agent_name="specialist",
        summary=summary,
        classification=EvidenceClassification.CURRENT,
    )


def _run(awaitable):
    return asyncio.run(awaitable)


def test_historical_node_returns_only_historical_partial_update(monkeypatch):
    update = {"historical_evidence": [], "rag_observations": [], "errors": []}
    calls = []

    def fake_agent(state, *, llm=None):
        calls.append((state, llm))
        return update

    monkeypatch.setattr(nodes, "analyze_historical_incidents", fake_agent)
    state = _state()
    before = copy.deepcopy(state)
    fake_llm = object()

    result = nodes.historical_rag_node(state, llm=fake_llm)

    assert result is update
    assert calls == [(state, fake_llm)]
    assert state == before


def test_each_operational_node_returns_its_agent_partial_update(monkeypatch):
    async def fake_agent(state):
        return {"source_evidence": [], "errors": []}

    monkeypatch.setattr(nodes, "analyze_logs", fake_agent)
    monkeypatch.setattr(nodes, "analyze_service_metrics", fake_agent)
    monkeypatch.setattr(nodes, "analyze_prometheus", fake_agent)
    monkeypatch.setattr(nodes, "analyze_grafana", fake_agent)
    state = _state()
    before = copy.deepcopy(state)

    async def run_nodes():
        return await asyncio.gather(
            nodes.log_analysis_node(state),
            nodes.metrics_analysis_node(state),
            nodes.prometheus_analysis_node(state),
            nodes.grafana_analysis_node(state),
        )

    results = _run(run_nodes())
    assert results == [{"source_evidence": [], "errors": []}] * 4
    assert state == before


def test_parallel_source_updates_merge_without_overwriting_other_branches():
    state = _state()
    historical = _evidence(
        "historical:1",
        EvidenceSourceType.HISTORICAL_RAG,
        EvidenceClassification.HISTORICAL,
    )
    log = _evidence(
        "log:1",
        EvidenceSourceType.APPLICATION_LOGS,
        EvidenceClassification.CURRENT,
    )
    metrics = _evidence(
        "metrics:1",
        EvidenceSourceType.SERVICE_METRICS,
        EvidenceClassification.CURRENT,
    )

    state["historical_evidence"] = merge_evidence(
        state["historical_evidence"], [historical]
    )
    state["log_evidence"] = merge_evidence(state["log_evidence"], [log])
    state["metrics_evidence"] = merge_evidence(state["metrics_evidence"], [metrics])

    assert state["historical_evidence"] == [historical]
    assert state["log_evidence"] == [log]
    assert state["metrics_evidence"] == [metrics]


def test_aggregation_deduplicates_evidence_and_keeps_history_classification():
    historical = _evidence(
        "historical-id",
        EvidenceSourceType.HISTORICAL_RAG,
        EvidenceClassification.HISTORICAL,
    )
    current = _evidence(
        "same-current-id",
        EvidenceSourceType.APPLICATION_LOGS,
        EvidenceClassification.CURRENT,
    )
    duplicate = current.model_copy()
    current = _evidence(
        "current-id",
        EvidenceSourceType.PROMETHEUS,
        EvidenceClassification.CURRENT,
    )
    state = _state()
    state["historical_evidence"] = [historical]
    state["log_evidence"] = [duplicate, duplicate]
    state["prometheus_evidence"] = [current]

    result = nodes.evidence_aggregation_node(state)

    assert result["consolidated_evidence"] == [historical, duplicate, current]
    assert (
        result["consolidated_evidence"][0].classification
        is EvidenceClassification.HISTORICAL
    )
    assert (
        result["consolidated_evidence"][1].classification
        is EvidenceClassification.CURRENT
    )
    assert state["historical_evidence"] == [historical]


def test_aggregation_preserves_conflicting_observations_gaps_errors_and_warnings():
    historical_observation = AgentObservation(
        observation_id="shared",
        agent_name="historical",
        summary="Prior event showed a transient failure.",
        classification=EvidenceClassification.HISTORICAL,
    )
    current_observation = _observation(
        "shared",
        "Current logs show a different failure pattern.",
    )
    gap = EvidenceGap(gap_id="gap-1", description="Metrics are unavailable.")
    state = _state()
    state["rag_observations"] = [historical_observation]
    state["observations"] = [current_observation]
    state["evidence_gaps"] = [gap]
    state["errors"] = ["Prometheus query failed."]
    state["warnings"] = ["Grafana returned no alerts."]

    result = nodes.evidence_aggregation_node(state)

    assert len(result["observations"]) == 2
    assert {item.summary for item in result["observations"]} == {
        historical_observation.summary,
        current_observation.summary,
    }
    assert len({item.observation_id for item in result["observations"]}) == 2
    assert result["evidence_gaps"] == [gap]
    assert result["errors"] == ["Prometheus query failed."]
    assert result["warnings"] == [
        "Grafana returned no alerts.",
        "Conflicting observations shared ID 'shared'; both were retained under distinct IDs.",
    ]
    assert (
        merge_observations(
            state["observations"],
            result["observations"],
        )
        == result["observations"]
    )


def test_aggregation_empty_evidence_is_an_empty_no_op():
    state = _state()
    state["evidence_gaps"] = [EvidenceGap(gap_id="gap-1", description="No logs.")]
    state["warnings"] = ["Log server returned no records."]

    result = nodes.evidence_aggregation_node(state)

    assert result["consolidated_evidence"] == []
    assert result["evidence_gaps"] == state["evidence_gaps"]
    assert result["warnings"] == state["warnings"]


def test_all_nodes_are_importable_independently():
    from app.graph.nodes import (
        evidence_aggregation_node,
        grafana_analysis_node,
        historical_rag_node,
        log_analysis_node,
        metrics_analysis_node,
        prometheus_analysis_node,
    )

    assert all(
        callable(node)
        for node in (
            evidence_aggregation_node,
            grafana_analysis_node,
            historical_rag_node,
            log_analysis_node,
            metrics_analysis_node,
            prometheus_analysis_node,
        )
    )


def test_observation_reducer_preserves_distinct_conflicting_observation_ids():
    first = _observation("observation-1", "Observation A.")
    second = _observation("observation-2", "Observation B.")

    assert merge_observations([], [first, second]) == [first, second]
