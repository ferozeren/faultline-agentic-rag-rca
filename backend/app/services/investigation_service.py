"""Adapter between validated incident requests and the existing LangGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.agent_utils import configured_nvidia_model
from app.schemas.evidence import Evidence
from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import InvestigationReport
from app.schemas.supervisor import InvestigationDecision


class InvestigationServiceError(RuntimeError):
    """Safe service failure with a stable API error code and public message."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


class InvestigationPersistenceError(RuntimeError):
    """Safe database persistence failure with a stable API response."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass(frozen=True)
class InvestigationServiceResult:
    """Validated report and supervisor decision extracted from the graph state."""

    report: InvestigationReport
    supervisor_decision: InvestigationDecision | None
    evidence: list[Evidence]
    execution_metadata: dict[str, Any]


class InvestigationService:
    """Invoke the public async graph interface and validate its final report."""

    _RCA_MODEL_CONFIGURATION_ERROR = (
        "NVIDIA RCA model configuration is incomplete."
    )

    async def investigate(
        self,
        incident: IncidentRequest,
    ) -> InvestigationServiceResult:
        try:
            from app.graph.workflow import run_investigation

            try:
                nvidia_llm = configured_nvidia_model(
                    error_message=self._RCA_MODEL_CONFIGURATION_ERROR,
                )
            except RuntimeError as error:
                if str(error) != self._RCA_MODEL_CONFIGURATION_ERROR:
                    raise
                nvidia_llm = None

            state = await run_investigation(
                incident,
                llm=nvidia_llm,
                supervisor_llm=nvidia_llm,
                rca_llm=nvidia_llm,
            )
        except Exception as error:
            raise InvestigationServiceError(
                "workflow_execution_failed",
                "The investigation workflow could not be completed.",
            ) from error

        if not isinstance(state, dict):
            raise InvestigationServiceError(
                "invalid_workflow_result",
                "The investigation workflow returned an invalid result.",
            )

        report_value = state.get("investigation_report")
        final_report = state.get("final_report")
        if report_value is None:
            report_value = final_report
        if report_value is None:
            raise InvestigationServiceError(
                "report_unavailable",
                "The investigation completed without a report.",
            )

        try:
            evidence_by_id: dict[str, Evidence] = {}
            for field in (
                "consolidated_evidence",
                "historical_evidence",
                "log_evidence",
                "metrics_evidence",
                "prometheus_evidence",
                "grafana_evidence",
            ):
                for candidate in state.get(field, []) or []:
                    evidence = (
                        candidate
                        if isinstance(candidate, Evidence)
                        else Evidence.model_validate(candidate)
                    )
                    previous = evidence_by_id.get(evidence.evidence_id)
                    if previous is not None and previous != evidence:
                        raise ValueError(
                            "Workflow state contains conflicting evidence IDs."
                        )
                    evidence_by_id[evidence.evidence_id] = evidence

            report_payload: Any = (
                report_value.model_dump(mode="python")
                if isinstance(report_value, InvestigationReport)
                else report_value
            )
            report = InvestigationReport.model_validate(
                report_payload,
                context={"evidence_by_id": evidence_by_id},
            )
            if final_report is not None and report_value is not final_report:
                final_payload = (
                    final_report.model_dump(mode="python")
                    if isinstance(final_report, InvestigationReport)
                    else final_report
                )
                final_validated = InvestigationReport.model_validate(
                    final_payload,
                    context={"evidence_by_id": evidence_by_id},
                )
                if final_validated != report:
                    raise ValueError("Workflow returned inconsistent reports.")
            decision_value = state.get("supervisor_decision")
            decision = (
                InvestigationDecision.model_validate(
                    decision_value.model_dump(mode="python")
                    if isinstance(decision_value, InvestigationDecision)
                    else decision_value
                )
                if decision_value is not None
                else None
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise InvestigationServiceError(
                "invalid_workflow_result",
                "The investigation workflow returned invalid report data.",
            ) from error

        return InvestigationServiceResult(
            report=report,
            supervisor_decision=decision,
            evidence=list(evidence_by_id.values()),
            execution_metadata=(
                state.get("execution_metadata", {})
                if isinstance(state.get("execution_metadata", {}), dict)
                else {}
            ),
        )
