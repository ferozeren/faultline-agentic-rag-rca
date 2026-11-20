"""Independently callable specialist and evidence aggregation nodes."""

from __future__ import annotations

from typing import Any

from app.agents.agent_utils import stable_id
from app.agents.grafana_analysis_agent import analyze_grafana
from app.agents.historical_rag_agent import analyze_historical_incidents
from app.agents.log_analysis_agent import analyze_logs
from app.agents.metrics_analysis_agent import analyze_service_metrics
from app.agents.prometheus_analysis_agent import analyze_prometheus
from app.agents.recommendation_agent import generate_recommendations
from app.agents.report_generation_agent import generate_investigation_report
from app.agents.root_cause_agent import analyze_root_cause
from app.agents.supervisor_agent import assess_investigation
from app.graph.state import InvestigationState
from app.schemas.evidence import Evidence, EvidenceClassification
from app.schemas.findings import AgentObservation
from app.schemas.investigation_report import InvestigationStatus, ReportStatus
from app.schemas.rca import RCAResult, RCAStatus
from app.schemas.recommendations import RecommendationResult, RecommendationStatus
from app.schemas.supervisor import DecisionStatus, InvestigationDecision


def historical_rag_node(
    state: InvestigationState,
    *,
    llm: Any | None = None,
) -> dict[str, object]:
    """Run only the historical retrieval branch."""
    return analyze_historical_incidents(state, llm=llm)


async def log_analysis_node(state: InvestigationState) -> dict[str, object]:
    """Run only the application-log branch."""
    return await analyze_logs(state)


async def metrics_analysis_node(state: InvestigationState) -> dict[str, object]:
    """Run only the service-metrics branch."""
    return await analyze_service_metrics(state)


async def prometheus_analysis_node(state: InvestigationState) -> dict[str, object]:
    """Run only the Prometheus branch."""
    return await analyze_prometheus(state)


async def grafana_analysis_node(state: InvestigationState) -> dict[str, object]:
    """Run only the Grafana branch."""
    return await analyze_grafana(state)


async def supervisor_node(
    state: InvestigationState,
    *,
    llm: Any | None = None,
) -> dict[str, object]:
    """Assess aggregated evidence and return only supervisor-owned state updates."""
    try:
        result = await assess_investigation(state, llm=llm)
    except Exception as error:  # noqa: BLE001 - preserve the investigation on supervisor failure.
        result = await assess_investigation(state, llm=None)
        metadata = next(iter(result.execution_metadata.values()))
        metadata["status"] = "failed"
        metadata["fallback_used"] = True
        metadata["error_type"] = type(error).__name__
        warning = (
            f"Supervisor failed ({type(error).__name__}); a deterministic fallback "
            "decision was used."
        )
        return {
            "supervisor_decision": result.decision,
            "execution_metadata": result.execution_metadata,
            "warnings": [*result.warnings, warning],
            "errors": [
                *result.errors,
                f"supervisor failed ({type(error).__name__}).",
            ],
        }
    return {
        "supervisor_decision": result.decision,
        "execution_metadata": result.execution_metadata,
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


async def root_cause_analysis_node(
    state: InvestigationState,
    *,
    llm: Any | None = None,
) -> dict[str, object]:
    """Run RCA only when a validated Supervisor decision is ready."""
    candidate = state.get("supervisor_decision")
    try:
        decision = (
            candidate
            if isinstance(candidate, InvestigationDecision)
            else InvestigationDecision.model_validate(candidate)
            if candidate is not None
            else None
        )
    except (TypeError, ValueError) as error:
        decision = None
        gate_warning = (
            f"RCA was skipped because the Supervisor decision is invalid "
            f"({type(error).__name__})."
        )
    else:
        gate_warning = (
            "RCA was skipped because the Supervisor decision was not ready."
            if decision is None
            or decision.decision_status is not DecisionStatus.READY_FOR_RCA
            else None
        )

    if gate_warning:
        result = RCAResult(
            status=RCAStatus.SKIPPED,
            incident_summary=state.get("incident_description", "Incident analysis"),
            overall_confidence=0,
            confidence_explanation=(
                "No RCA confidence was assigned because the Supervisor gate was "
                "not satisfied."
            ),
            analysis_summary=gate_warning,
            limitations=[gate_warning],
        )
        return {
            "rca_result": result,
            "warnings": [gate_warning],
        }

    try:
        agent_result = await analyze_root_cause(state, llm=llm)
    except Exception as error:  # noqa: BLE001 - preserve preceding investigation results.
        warning = (
            f"RCA analysis failed ({type(error).__name__}); no cause was asserted."
        )
        result = RCAResult(
            status=RCAStatus.FAILED,
            incident_summary=state.get("incident_description", "Incident analysis"),
            overall_confidence=0,
            confidence_explanation=(
                "No primary cause was assigned because RCA analysis failed."
            ),
            analysis_summary=warning,
            limitations=[warning],
        )
        return {
            "rca_result": result,
            "execution_metadata": {},
            "warnings": [warning],
            "errors": [f"RCA analysis failed ({type(error).__name__})."],
        }
    return {
        "rca_result": agent_result.rca_result,
        "execution_metadata": agent_result.execution_metadata,
        "warnings": list(agent_result.warnings),
        "errors": list(agent_result.errors),
    }


async def recommendation_node(
    state: InvestigationState,
    *,
    llm: Any | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Generate recommendations from a valid, analyzed RCA finding."""
    try:
        result = await generate_recommendations(
            state,
            llm=llm,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:  # noqa: BLE001 - retain investigation state even on recommendation failures.
        error_message = (
            f"Recommendation analysis failed ({type(error).__name__}); "
            "no actions were generated."
        )
        failure = RecommendationResult(
            status=RecommendationStatus.FAILED,
            incident_id=state.get("incident_id"),
            overall_summary="Recommendation generation failed; no actions were generated.",
            errors=[error_message],
        )
        return {
            "recommendation_result": failure,
            "execution_metadata": {
                "recommendation_analysis": {
                    "agent_name": "recommendation_analysis",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "evidence_count": 0,
                }
            },
            "warnings": [],
            "errors": [error_message],
        }

    return {
        "recommendation_result": result,
        "execution_metadata": {
            "recommendation_analysis": {
                "agent_name": "recommendation_analysis",
                "status": (
                    "success"
                    if result.status is RecommendationStatus.COMPLETED
                    else "failed"
                    if result.status is RecommendationStatus.FAILED
                    else "skipped"
                ),
                "evidence_count": sum(
                    len(item.supporting_evidence_ids) for item in result.recommendations
                ),
                "recommended_action_count": len(result.recommendations),
            }
        },
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


async def report_generation_node(
    state: InvestigationState,
    *,
    llm: Any | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Assemble a report from state without changing the underlying findings."""
    report = await generate_investigation_report(
        state,
        llm=llm,
        timeout_seconds=timeout_seconds,
    )
    status = report.report_status
    execution_status = (
        "success"
        if status is ReportStatus.COMPLETED
        else "failed"
        if status is ReportStatus.FAILED
        else "partial"
    )
    return {
        "investigation_report": report,
        "final_report": report,
        "execution_metadata": {
            "report_generation": {
                "agent_name": "report_generation",
                "status": execution_status,
                "evidence_count": len(report.evidence_references),
                "finding_count": int(
                    report.root_cause_analysis is not None
                    and report.root_cause_analysis.primary_finding is not None
                ),
                "recommendation_count": (
                    len(report.recommendation_result.recommendations)
                    if report.recommendation_result
                    else 0
                ),
            }
        },
        "warnings": report.warnings,
        "errors": report.errors,
    }


def evidence_aggregation_node(
    state: InvestigationState,
) -> dict[str, object]:
    """Validate and combine branch evidence without inferring root cause."""
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    evidence_by_id: dict[str, Evidence] = {}
    observations_by_id: dict[str, AgentObservation] = {}
    observations: list[AgentObservation] = [
        *state.get("observations", []),
        *state.get("rag_observations", []),
    ]

    evidence_channels = (
        ("historical_evidence", EvidenceClassification.HISTORICAL),
        ("log_evidence", EvidenceClassification.CURRENT),
        ("metrics_evidence", EvidenceClassification.CURRENT),
        ("prometheus_evidence", EvidenceClassification.CURRENT),
        ("grafana_evidence", EvidenceClassification.CURRENT),
    )
    for channel, expected_classification in evidence_channels:
        for candidate in state.get(channel, []):
            try:
                payload = (
                    candidate.model_dump(mode="python")
                    if isinstance(candidate, Evidence)
                    else candidate
                )
                evidence = Evidence.model_validate(payload)
            except ValueError:
                errors.append(
                    f"Evidence aggregation rejected an invalid item from {channel}."
                )
                continue
            if evidence.classification is not expected_classification:
                warnings.append(
                    f"Evidence {evidence.evidence_id!r} from {channel} had an "
                    "unexpected classification and was excluded."
                )
                continue
            existing = evidence_by_id.get(evidence.evidence_id)
            if existing is None:
                evidence_by_id[evidence.evidence_id] = evidence
            elif existing != evidence:
                warnings.append(
                    f"Conflicting payloads shared evidence ID "
                    f"{evidence.evidence_id!r}; the first evidence item was retained."
                )

    for observation in observations:
        existing = observations_by_id.get(observation.observation_id)
        if existing is None:
            observations_by_id[observation.observation_id] = observation
        elif existing != observation:
            replacement_id = stable_id(
                f"observation-conflict:{observation.observation_id}",
                observation.model_dump(mode="json"),
            )
            observations_by_id[replacement_id] = observation.model_copy(
                update={"observation_id": replacement_id}
            )
            warnings.append(
                f"Conflicting observations shared ID {observation.observation_id!r}; "
                "both were retained under distinct IDs."
            )

    consolidated = list(evidence_by_id.values())
    branch_statuses = [
        value.get("status")
        for value in state.get("execution_metadata", {}).values()
        if isinstance(value, dict) and isinstance(value.get("status"), str)
    ]
    failed_branches = sum(status == "failed" for status in branch_statuses)
    incomplete_branches = any(
        status in {"failed", "partial_success", "no_evidence"}
        for status in branch_statuses
    )
    if consolidated:
        status = (
            InvestigationStatus.PARTIAL_SUCCESS
            if incomplete_branches or errors
            else InvestigationStatus.SUCCESS
        )
    elif branch_statuses and failed_branches == len(branch_statuses):
        status = InvestigationStatus.FAILED
    elif failed_branches or errors:
        status = InvestigationStatus.PARTIAL_SUCCESS
    else:
        status = InvestigationStatus.NO_EVIDENCE

    return {
        "consolidated_evidence": consolidated,
        "observations": list(observations_by_id.values()),
        "evidence_gaps": list(state.get("evidence_gaps", [])),
        "investigation_status": status,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors)),
    }
