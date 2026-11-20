"""Shared LangGraph investigation state and idempotent parallel reducers.

Each evidence source has its own channel so the historical RAG branch and
operational MCP branch cannot replace one another's evidence. List channels use
stable identifiers: exact retries are ignored, conflicting values under an
existing identifier raise an error, and empty branch updates leave prior values
unchanged. Match updates are deduplicated by RAG chunk ID. Status, stage,
selected finding, RCA result, recommendation result, and final report remain
single-writer fields. Execution metadata merges by key and rejects conflicting
values. Both report fields are single-writer: ``investigation_report`` is the
structured report, while ``final_report`` remains its compatibility alias.
"""

from collections.abc import Callable, Hashable
from datetime import datetime
from typing import Annotated, TypedDict, TypeVar

from pydantic import JsonValue

from app.schemas.evidence import Evidence
from app.schemas.findings import (
    AgentObservation,
    EvidenceGap,
    Recommendation,
    RootCauseFinding,
)
from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import InvestigationReport, InvestigationStatus
from app.schemas.rca import RCAResult
from app.schemas.recommendations import RecommendationResult
from app.schemas.supervisor import InvestigationDecision

T = TypeVar("T")


def _merge_by_key(
    current: list[T],
    updates: list[T],
    *,
    key: Callable[[T], Hashable],
    item_name: str,
) -> list[T]:
    merged = list(current)
    positions = {key(item): index for index, item in enumerate(merged)}

    for item in updates:
        item_key = key(item)
        existing_index = positions.get(item_key)
        if existing_index is None:
            positions[item_key] = len(merged)
            merged.append(item)
        elif merged[existing_index] != item:
            raise ValueError(
                f"Conflicting {item_name} updates for identifier {item_key!r}."
            )
    return merged


def merge_evidence(current: list[Evidence], updates: list[Evidence]) -> list[Evidence]:
    """Merge evidence idempotently, retaining source attribution in its key."""
    return _merge_by_key(
        current,
        updates,
        key=lambda item: (item.source_type, item.evidence_id),
        item_name="evidence",
    )


def merge_historical_matches(
    current: list[dict[str, JsonValue]],
    updates: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Merge RAG chunks once by their stable chunk ID."""

    def chunk_id(match: dict[str, JsonValue]) -> str:
        value = match.get("chunk_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Historical incident matches require a chunk_id.")
        return value

    return _merge_by_key(
        current,
        updates,
        key=chunk_id,
        item_name="historical match",
    )


def merge_observations(
    current: list[AgentObservation],
    updates: list[AgentObservation],
) -> list[AgentObservation]:
    return _merge_by_key(
        current,
        updates,
        key=lambda item: item.observation_id,
        item_name="observation",
    )


def merge_evidence_gaps(
    current: list[EvidenceGap],
    updates: list[EvidenceGap],
) -> list[EvidenceGap]:
    return _merge_by_key(
        current,
        updates,
        key=lambda item: item.gap_id,
        item_name="evidence gap",
    )


def merge_root_cause_findings(
    current: list[RootCauseFinding],
    updates: list[RootCauseFinding],
) -> list[RootCauseFinding]:
    return _merge_by_key(
        current,
        updates,
        key=lambda item: item.finding_id,
        item_name="root-cause finding",
    )


def merge_recommendations(
    current: list[Recommendation],
    updates: list[Recommendation],
) -> list[Recommendation]:
    return _merge_by_key(
        current,
        updates,
        key=lambda item: item.recommendation_id,
        item_name="recommendation",
    )


def merge_messages(current: list[str], updates: list[str]) -> list[str]:
    """Merge error/warning messages once, preserving first-seen ordering."""
    return _merge_by_key(
        current,
        updates,
        key=lambda message: message,
        item_name="message",
    )


def merge_execution_metadata(
    current: dict[str, JsonValue],
    updates: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Merge branch metadata by key without hiding conflicting values."""
    merged = dict(current)
    for key, value in updates.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting execution metadata for key {key!r}.")
        merged[key] = value
    return merged


class InvestigationState(TypedDict, total=False):
    """Typed state passed between investigation graph nodes."""

    incident_id: str | None
    incident_description: str
    affected_service: str | None
    environment: str | None
    severity: str | None
    start_time: datetime | None
    end_time: datetime | None
    additional_context: dict[str, JsonValue] | None

    investigation_status: InvestigationStatus
    current_stage: str | None
    execution_metadata: Annotated[dict[str, JsonValue], merge_execution_metadata]

    historical_evidence: Annotated[list[Evidence], merge_evidence]
    historical_incident_matches: Annotated[
        list[dict[str, JsonValue]], merge_historical_matches
    ]
    rag_observations: Annotated[list[AgentObservation], merge_observations]

    log_evidence: Annotated[list[Evidence], merge_evidence]
    metrics_evidence: Annotated[list[Evidence], merge_evidence]
    prometheus_evidence: Annotated[list[Evidence], merge_evidence]
    grafana_evidence: Annotated[list[Evidence], merge_evidence]
    consolidated_evidence: Annotated[list[Evidence], merge_evidence]

    observations: Annotated[list[AgentObservation], merge_observations]
    evidence_gaps: Annotated[list[EvidenceGap], merge_evidence_gaps]
    root_cause_findings: Annotated[list[RootCauseFinding], merge_root_cause_findings]
    selected_root_cause: RootCauseFinding | None
    recommendations: Annotated[list[Recommendation], merge_recommendations]

    final_report: InvestigationReport | None
    investigation_report: InvestigationReport | None
    supervisor_decision: InvestigationDecision | None
    rca_result: RCAResult | None
    recommendation_result: RecommendationResult | None
    errors: Annotated[list[str], merge_messages]
    warnings: Annotated[list[str], merge_messages]


def create_initial_state(request: IncidentRequest) -> InvestigationState:
    """Create a complete, empty-evidence state from validated incident input."""
    return {
        "incident_id": request.incident_id,
        "incident_description": request.description,
        "affected_service": request.affected_service,
        "environment": request.environment,
        "severity": request.severity,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "additional_context": request.additional_context,
        "investigation_status": InvestigationStatus.PENDING,
        "current_stage": None,
        "execution_metadata": {},
        "historical_evidence": [],
        "historical_incident_matches": [],
        "rag_observations": [],
        "log_evidence": [],
        "metrics_evidence": [],
        "prometheus_evidence": [],
        "grafana_evidence": [],
        "consolidated_evidence": [],
        "observations": [],
        "evidence_gaps": [],
        "root_cause_findings": [],
        "selected_root_cause": None,
        "recommendations": [],
        "final_report": None,
        "investigation_report": None,
        "supervisor_decision": None,
        "rca_result": None,
        "recommendation_result": None,
        "errors": [],
        "warnings": [],
    }
