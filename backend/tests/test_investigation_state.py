"""Tests for state initialization and idempotent parallel reducers."""

from datetime import UTC, datetime
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from app.graph import InvestigationState, create_initial_state
from app.graph.state import (
    merge_evidence,
    merge_evidence_gaps,
    merge_execution_metadata,
    merge_historical_matches,
    merge_messages,
    merge_observations,
    merge_recommendations,
    merge_root_cause_findings,
)
from app.schemas import (
    AgentObservation,
    Evidence,
    EvidenceClassification,
    EvidenceGap,
    EvidenceSourceType,
    IncidentRequest,
    Recommendation,
    RootCauseFinding,
)
from langgraph.graph import StateGraph


def _evidence(source_type: EvidenceSourceType, evidence_id: str) -> Evidence:
    classification = (
        EvidenceClassification.HISTORICAL
        if source_type is EvidenceSourceType.HISTORICAL_RAG
        else EvidenceClassification.CURRENT
    )
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description=f"Evidence from {source_type.value}.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        classification=classification,
    )


def test_state_initialization_includes_all_empty_channels():
    state = create_initial_state(
        IncidentRequest(
            incident_id="INC-18",
            description="Elevated latency",
            affected_service="checkout",
            environment="production",
            severity="high",
        )
    )

    assert state["incident_id"] == "INC-18"
    assert state["incident_description"] == "Elevated latency"
    assert state["investigation_status"].value == "pending"
    assert state["execution_metadata"] == {}
    for field in (
        "historical_evidence",
        "historical_incident_matches",
        "rag_observations",
        "log_evidence",
        "metrics_evidence",
        "prometheus_evidence",
        "grafana_evidence",
        "consolidated_evidence",
        "observations",
        "evidence_gaps",
        "root_cause_findings",
        "recommendations",
        "errors",
        "warnings",
    ):
        assert state[field] == []
    assert state["selected_root_cause"] is None
    assert state["final_report"] is None
    assert state["investigation_report"] is None
    assert state["rca_result"] is None
    assert state["recommendation_result"] is None


def test_historical_rag_state_update_preserves_chunk_and_evidence_ids():
    evidence = _evidence(EvidenceSourceType.HISTORICAL_RAG, "chunk-18")
    match = {
        "chunk_id": "chunk-18",
        "incident_id": "INC-7",
        "text": "A prior connection saturation incident.",
    }

    state = create_initial_state(IncidentRequest(description="Elevated latency"))
    state["historical_evidence"] = merge_evidence(
        state["historical_evidence"], [evidence]
    )
    state["historical_incident_matches"] = merge_historical_matches(
        state["historical_incident_matches"], [match]
    )

    assert state["historical_evidence"] == [evidence]
    assert state["historical_incident_matches"] == [match]


@pytest.mark.parametrize(
    ("field", "source_type"),
    [
        ("log_evidence", EvidenceSourceType.APPLICATION_LOGS),
        ("metrics_evidence", EvidenceSourceType.SERVICE_METRICS),
        ("prometheus_evidence", EvidenceSourceType.PROMETHEUS),
        ("grafana_evidence", EvidenceSourceType.GRAFANA),
    ],
)
def test_operational_source_state_updates(field, source_type):
    state = create_initial_state(IncidentRequest(description="Elevated latency"))
    evidence = _evidence(source_type, f"{source_type.value}-18")

    state[field] = merge_evidence(state[field], [evidence])

    assert state[field] == [evidence]
    assert state[field][0].source_type is source_type
    assert state[field][0].classification is EvidenceClassification.CURRENT


def test_parallel_evidence_updates_merge_without_overwriting_or_duplication():
    historical = _evidence(EvidenceSourceType.HISTORICAL_RAG, "shared-id")
    log = _evidence(EvidenceSourceType.APPLICATION_LOGS, "shared-id")

    historical_result = merge_evidence([], [historical])
    log_result = merge_evidence([], [log])
    merged = merge_evidence(historical_result, [log, historical])

    assert log_result == [log]
    assert merged == [historical, log]
    assert [item.source_type for item in merged] == [
        EvidenceSourceType.HISTORICAL_RAG,
        EvidenceSourceType.APPLICATION_LOGS,
    ]


def test_duplicate_ids_with_conflicting_evidence_fail_explicitly():
    original = _evidence(EvidenceSourceType.PROMETHEUS, "query-1")
    conflicting = original.model_copy(update={"description": "Different payload."})

    with pytest.raises(ValueError, match="Conflicting evidence"):
        merge_evidence([original], [conflicting])


def test_empty_evidence_update_is_a_no_op():
    evidence = _evidence(EvidenceSourceType.GRAFANA, "dashboard-1")

    assert merge_evidence([], []) == []
    assert merge_evidence([evidence], []) == [evidence]


def test_historical_match_retries_are_idempotent_and_require_chunk_ids():
    match = {"chunk_id": "chunk-1", "text": "Prior incident."}

    assert merge_historical_matches([match], [match]) == [match]
    with pytest.raises(ValueError, match="require a chunk_id"):
        merge_historical_matches([], [{"text": "Missing stable ID"}])


def test_observation_finding_gap_and_recommendation_reducers_are_idempotent():
    observation = AgentObservation(
        observation_id="obs-1",
        agent_name="log_analysis",
        summary="One observation.",
        classification=EvidenceClassification.CURRENT,
    )
    gap = EvidenceGap(gap_id="gap-1", description="No traces.")
    finding = RootCauseFinding(
        finding_id="cause-1",
        probable_cause="Connection saturation",
        confidence=0.5,
        explanation="A possible cause.",
    )
    recommendation = Recommendation(
        recommendation_id="rec-1",
        action="Review pool limits.",
        rationale="Connections may be saturated.",
        priority="medium",
        expected_outcome="Confirm the source of saturation.",
    )

    assert merge_observations([observation], [observation]) == [observation]
    assert merge_evidence_gaps([gap], [gap]) == [gap]
    assert merge_root_cause_findings([finding], [finding]) == [finding]
    assert merge_recommendations([recommendation], [recommendation]) == [recommendation]


def test_error_and_warning_updates_merge_without_repeated_messages():
    assert merge_messages(["timeout"], ["timeout", "source unavailable"]) == [
        "timeout",
        "source unavailable",
    ]


def test_execution_metadata_merges_branch_keys_and_rejects_conflicts():
    merged = merge_execution_metadata(
        {"historical_retrieval_count": 3},
        {"log_query_count": 2, "historical_retrieval_count": 3},
    )

    assert merged == {"historical_retrieval_count": 3, "log_query_count": 2}
    with pytest.raises(ValueError, match="Conflicting execution metadata"):
        merge_execution_metadata(merged, {"log_query_count": 4})


def test_annotated_state_channels_declare_langgraph_reducers():
    annotations = get_type_hints(InvestigationState, include_extras=True)

    for field, reducer in (
        ("execution_metadata", merge_execution_metadata),
        ("historical_evidence", merge_evidence),
        ("historical_incident_matches", merge_historical_matches),
        ("rag_observations", merge_observations),
        ("log_evidence", merge_evidence),
        ("metrics_evidence", merge_evidence),
        ("prometheus_evidence", merge_evidence),
        ("grafana_evidence", merge_evidence),
        ("consolidated_evidence", merge_evidence),
        ("observations", merge_observations),
        ("evidence_gaps", merge_evidence_gaps),
        ("root_cause_findings", merge_root_cause_findings),
        ("recommendations", merge_recommendations),
        ("errors", merge_messages),
        ("warnings", merge_messages),
    ):
        annotation = annotations[field]
        assert get_origin(annotation) is Annotated
        assert get_args(annotation)[1] is reducer


def test_langgraph_builds_aggregate_channels_from_state_annotations():
    builder = StateGraph(InvestigationState)

    assert builder.channels["historical_evidence"].operator is merge_evidence
    assert builder.channels["log_evidence"].operator is merge_evidence
    assert builder.channels["consolidated_evidence"].operator is merge_evidence
    assert builder.channels["observations"].operator is merge_observations
    assert builder.channels["warnings"].operator is merge_messages
    assert builder.channels["execution_metadata"].operator is merge_execution_metadata


def test_state_and_schema_import_compatibility():
    from app.graph import InvestigationState as ExportedState
    from app.schemas import IncidentRequest as ExportedRequest

    assert ExportedState is InvestigationState
    assert ExportedRequest is IncidentRequest
