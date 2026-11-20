"""Service tests for invoking the existing async workflow exactly once."""

import asyncio

import pytest

from app.graph import workflow
from app.schemas import IncidentRequest
from app.schemas.investigation_report import (
    InvestigationReport,
    InvestigationStatus,
    ReportStatus,
)
from app.services.investigation_service import (
    InvestigationService,
    InvestigationServiceError,
)
import app.services.investigation_service as investigation_service


@pytest.fixture(autouse=True)
def mock_rca_model_factory(monkeypatch):
    monkeypatch.setattr(
        investigation_service,
        "configured_nvidia_model",
        lambda **_kwargs: None,
    )


def _incident():
    return IncidentRequest(
        incident_id="INC-SERVICE",
        description="Elevated checkout latency.",
        affected_service="checkout",
    )


def _report():
    return InvestigationReport(
        incident_id="INC-SERVICE",
        incident_summary="Elevated checkout latency.",
        affected_service="checkout",
        investigation_status=InvestigationStatus.PARTIAL,
        report_status=ReportStatus.PARTIAL,
    )


def test_service_invokes_async_workflow_once_and_validates_report(monkeypatch):
    calls = []
    report = _report()

    async def fake_workflow(incident, *, llm, supervisor_llm, rca_llm):
        calls.append((incident, llm, supervisor_llm, rca_llm))
        return {
            "investigation_report": report,
            "final_report": report,
            "supervisor_decision": None,
        }

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    result = asyncio.run(InvestigationService().investigate(_incident()))

    assert result.report == report
    assert result.supervisor_decision is None
    assert len(calls) == 1
    assert calls[0][0].incident_id == "INC-SERVICE"
    assert calls[0][1:] == (None, None, None)


def test_service_passes_configured_nvidia_model_to_supervisor_and_rca(monkeypatch):
    report = _report()
    model = object()
    received_models = []
    monkeypatch.setattr(
        investigation_service,
        "configured_nvidia_model",
        lambda **_kwargs: model,
    )

    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        received_models.append((llm, supervisor_llm, rca_llm))
        return {"investigation_report": report}

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    result = asyncio.run(InvestigationService().investigate(_incident()))

    assert result.report == report
    assert received_models == [(model, model, model)]


def test_service_keeps_safe_fallback_when_nvidia_configuration_is_missing(monkeypatch):
    report = _report()
    missing_configuration = "NVIDIA RCA model configuration is incomplete."
    received_models = []

    def build_model(*, error_message):
        raise RuntimeError(error_message)

    monkeypatch.setattr(investigation_service, "configured_nvidia_model", build_model)

    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        received_models.append((llm, supervisor_llm, rca_llm))
        return {"investigation_report": report}

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    result = asyncio.run(InvestigationService().investigate(_incident()))

    assert result.report == report
    assert received_models == [(None, None, None)]


def test_service_accepts_legacy_final_report_field(monkeypatch):
    report = _report()

    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        return {"final_report": report}

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    result = asyncio.run(InvestigationService().investigate(_incident()))

    assert result.report == report


def test_service_fails_when_workflow_does_not_return_a_report(monkeypatch):
    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        return {"investigation_report": None, "final_report": None}

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)

    try:
        asyncio.run(InvestigationService().investigate(_incident()))
    except InvestigationServiceError as error:
        assert error.code == "report_unavailable"
    else:
        raise AssertionError("Missing reports must be a controlled service failure.")


def test_service_redacts_workflow_exception(monkeypatch):
    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        raise RuntimeError("token=private")

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)

    try:
        asyncio.run(InvestigationService().investigate(_incident()))
    except InvestigationServiceError as error:
        assert error.code == "workflow_execution_failed"
        assert "private" not in str(error)
    else:
        raise AssertionError("Workflow failures must be surfaced as service errors.")


def test_service_rejects_invalid_report_data(monkeypatch):
    async def fake_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        return {
            "investigation_report": {
                "incident_summary": "Invalid",
                "investigation_status": "unknown",
            }
        }

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)

    try:
        asyncio.run(InvestigationService().investigate(_incident()))
    except InvestigationServiceError as error:
        assert error.code == "invalid_workflow_result"
    else:
        raise AssertionError("Invalid reports must be rejected.")
