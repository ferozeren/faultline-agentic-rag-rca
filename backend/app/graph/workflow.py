"""Executable parallel LangGraph workflow for incident investigations."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.agent_utils import failed_agent_update
from app.graph import nodes
from app.graph.state import InvestigationState, create_initial_state
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import InvestigationStatus
from app.schemas.rca import RCAResult, RCAStatus
from app.schemas.supervisor import DecisionStatus, InvestigationDecision

SPECIALIST_NODE_NAMES = (
    "historical_rag",
    "log_analysis",
    "metrics_analysis",
    "prometheus_analysis",
    "grafana_analysis",
)


def _failed_update(
    *,
    agent_name: str,
    source_type: EvidenceSourceType,
    evidence_field: str,
    error: Exception,
    historical: bool = False,
) -> dict[str, object]:
    return failed_agent_update(
        agent_name=agent_name,
        source_type=source_type,
        evidence_field=evidence_field,
        error=error,
        historical=historical,
    )


async def _historical_rag_branch(
    state: InvestigationState,
    config: RunnableConfig,
) -> dict[str, object]:
    try:
        llm = config.get("configurable", {}).get("historical_rag_llm")
        if llm is None:
            result = await asyncio.to_thread(nodes.historical_rag_node, state)
        else:
            result = await asyncio.to_thread(
                nodes.historical_rag_node,
                state,
                llm=llm,
            )
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as error:  # noqa: BLE001 - report unexpected branch failures.
        return _failed_update(
            agent_name="historical_rag",
            source_type=EvidenceSourceType.HISTORICAL_RAG,
            evidence_field="historical_evidence",
            error=error,
            historical=True,
        )


async def _log_analysis_branch(state: InvestigationState) -> dict[str, object]:
    try:
        return await nodes.log_analysis_node(state)
    except Exception as error:  # noqa: BLE001 - report unexpected branch failures.
        return _failed_update(
            agent_name="log_analysis",
            source_type=EvidenceSourceType.APPLICATION_LOGS,
            evidence_field="log_evidence",
            error=error,
        )


async def _metrics_analysis_branch(state: InvestigationState) -> dict[str, object]:
    try:
        return await nodes.metrics_analysis_node(state)
    except Exception as error:  # noqa: BLE001 - report unexpected branch failures.
        return _failed_update(
            agent_name="metrics_analysis",
            source_type=EvidenceSourceType.SERVICE_METRICS,
            evidence_field="metrics_evidence",
            error=error,
        )


async def _prometheus_analysis_branch(state: InvestigationState) -> dict[str, object]:
    try:
        return await nodes.prometheus_analysis_node(state)
    except Exception as error:  # noqa: BLE001 - report unexpected branch failures.
        return _failed_update(
            agent_name="prometheus_analysis",
            source_type=EvidenceSourceType.PROMETHEUS,
            evidence_field="prometheus_evidence",
            error=error,
        )


async def _grafana_analysis_branch(state: InvestigationState) -> dict[str, object]:
    try:
        return await nodes.grafana_analysis_node(state)
    except Exception as error:  # noqa: BLE001 - report unexpected branch failures.
        return _failed_update(
            agent_name="grafana_analysis",
            source_type=EvidenceSourceType.GRAFANA,
            evidence_field="grafana_evidence",
            error=error,
        )


async def _supervisor_branch(
    state: InvestigationState,
    config: RunnableConfig,
) -> dict[str, object]:
    llm = config.get("configurable", {}).get("supervisor_llm")
    try:
        if llm is None:
            return await nodes.supervisor_node(state)
        return await nodes.supervisor_node(state, llm=llm)
    except Exception as error:  # noqa: BLE001 - preserve partial results for reporting.
        message = f"Supervisor analysis failed ({type(error).__name__})."
        return {
            "supervisor_decision": None,
            "execution_metadata": {
                "supervisor": {
                    "agent_name": "supervisor",
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            },
            "warnings": [message],
            "errors": [message],
        }


async def _root_cause_branch(
    state: InvestigationState,
    config: RunnableConfig,
) -> dict[str, object]:
    llm = config.get("configurable", {}).get("rca_llm")
    if llm is None:
        return await nodes.root_cause_analysis_node(state)
    return await nodes.root_cause_analysis_node(state, llm=llm)


async def _recommendation_branch(
    state: InvestigationState,
    config: RunnableConfig,
) -> dict[str, object]:
    llm = config.get("configurable", {}).get("recommendation_llm")
    if llm is None:
        return await nodes.recommendation_node(state)
    return await nodes.recommendation_node(state, llm=llm)


async def _report_generation_branch(
    state: InvestigationState,
    config: RunnableConfig,
) -> dict[str, object]:
    llm = config.get("configurable", {}).get("report_llm")
    if llm is None:
        return await nodes.report_generation_node(state)
    return await nodes.report_generation_node(state, llm=llm)


def _route_after_supervisor(state: InvestigationState) -> str:
    """Run RCA only for a valid ready decision; report all other outcomes."""
    candidate = state.get("supervisor_decision")
    try:
        decision = (
            candidate
            if isinstance(candidate, InvestigationDecision)
            else InvestigationDecision.model_validate(candidate)
            if candidate is not None
            else None
        )
    except (TypeError, ValueError):
        return "report_generation"
    if decision and decision.decision_status is DecisionStatus.READY_FOR_RCA:
        return "root_cause_analysis"
    return "report_generation"


def _route_after_rca(state: InvestigationState) -> str:
    """Run recommendations only for a valid analyzed RCA with a primary finding."""
    candidate = state.get("rca_result")
    evidence_by_id: dict[str, Evidence] = {}
    conflicting_ids: set[str] = set()
    for field in (
        "consolidated_evidence",
        "historical_evidence",
        "log_evidence",
        "metrics_evidence",
        "prometheus_evidence",
        "grafana_evidence",
    ):
        for item in state.get(field, []) or []:
            try:
                evidence = (
                    item
                    if isinstance(item, Evidence)
                    else Evidence.model_validate(item)
                )
            except (TypeError, ValueError):
                continue
            previous = evidence_by_id.get(evidence.evidence_id)
            if previous is not None and previous != evidence:
                conflicting_ids.add(evidence.evidence_id)
            else:
                evidence_by_id[evidence.evidence_id] = evidence
    try:
        payload = (
            candidate.model_dump(mode="python")
            if isinstance(candidate, RCAResult)
            else candidate
        )
        result = RCAResult.model_validate(
            payload,
            context={
                "evidence_by_id": evidence_by_id,
                "incident_service": state.get("affected_service"),
            },
        )
    except (TypeError, ValueError):
        return "report_generation"
    if (
        result.status is RCAStatus.ANALYZED
        and result.primary_finding is not None
        and not (set(result.primary_finding.supporting_evidence_ids) & conflicting_ids)
    ):
        return "recommendation"
    return "report_generation"


def build_investigation_graph() -> CompiledStateGraph:
    """Compile the specialist fan-out, gated analysis, and report terminal."""
    builder = StateGraph(InvestigationState)
    builder.add_node("historical_rag", _historical_rag_branch)
    builder.add_node("log_analysis", _log_analysis_branch)
    builder.add_node("metrics_analysis", _metrics_analysis_branch)
    builder.add_node("prometheus_analysis", _prometheus_analysis_branch)
    builder.add_node("grafana_analysis", _grafana_analysis_branch)
    builder.add_node("evidence_aggregation", nodes.evidence_aggregation_node)
    builder.add_node("supervisor", _supervisor_branch)
    builder.add_node("root_cause_analysis", _root_cause_branch)
    builder.add_node("recommendation", _recommendation_branch)
    builder.add_node("report_generation", _report_generation_branch)

    for node_name in SPECIALIST_NODE_NAMES:
        builder.add_edge(START, node_name)
    builder.add_edge(list(SPECIALIST_NODE_NAMES), "evidence_aggregation")
    builder.add_edge("evidence_aggregation", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "root_cause_analysis": "root_cause_analysis",
            "report_generation": "report_generation",
        },
    )
    builder.add_conditional_edges(
        "root_cause_analysis",
        _route_after_rca,
        {
            "recommendation": "recommendation",
            "report_generation": "report_generation",
        },
    )
    builder.add_edge("recommendation", "report_generation")
    builder.add_edge("report_generation", END)
    return builder.compile()


_investigation_graph = build_investigation_graph()


async def run_investigation(
    request: IncidentRequest,
    *,
    llm: Any | None = None,
    supervisor_llm: Any | None = None,
    rca_llm: Any | None = None,
    recommendation_llm: Any | None = None,
    report_llm: Any | None = None,
) -> InvestigationState:
    """Validate an incident, execute all branches, and return the merged state."""
    validated_request = IncidentRequest.model_validate(request).model_copy(deep=True)
    initial_state = create_initial_state(validated_request)
    initial_state["investigation_status"] = InvestigationStatus.IN_PROGRESS
    return await _investigation_graph.ainvoke(
        initial_state,
        config={
            "configurable": {
                "historical_rag_llm": llm,
                "supervisor_llm": supervisor_llm,
                "rca_llm": rca_llm,
                "recommendation_llm": recommendation_llm,
                "report_llm": report_llm,
            }
        },
    )


def get_investigation_graph_diagram() -> str:
    """Return a Mermaid diagram of the compiled fan-out and aggregation graph."""
    return (
        _investigation_graph.get_graph()
        .draw_mermaid()
        .replace("__start__", "START")
        .replace("__end__", "END")
    )
