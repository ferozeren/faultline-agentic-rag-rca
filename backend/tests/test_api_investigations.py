"""HTTP investigation endpoint tests with a mocked LangGraph workflow."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_database_session
from app.graph import workflow
from app.main import create_app
from app.schemas import (
    DecisionStatus,
    EvidenceSufficiency,
    HistoricalCurrentAlignment,
    IncidentRequest,
    InvestigationDecision,
    InvestigationEvidenceSummary,
    RecommendedNextStage,
)
from app.schemas.investigation_report import (
    InvestigationReport,
    InvestigationStatus,
    ReportStatus,
)
from fastapi.testclient import TestClient
import app.services.investigation_service as investigation_service
from app.database import repository

_TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(_TEST_ENGINE)
_TEST_SESSION = sessionmaker(bind=_TEST_ENGINE, class_=Session, expire_on_commit=False)


def _client():
    app = create_app()

    def override_session():
        with _TEST_SESSION() as session:
            yield session

    app.dependency_overrides[get_database_session] = override_session
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_rca_model_factory(monkeypatch):
    Base.metadata.drop_all(_TEST_ENGINE)
    Base.metadata.create_all(_TEST_ENGINE)
    monkeypatch.setattr(
        investigation_service,
        "configured_nvidia_model",
        lambda **_kwargs: None,
    )


def _report(*, status=ReportStatus.PARTIAL, incident_id="INC-API"):
    return InvestigationReport(
        incident_id=incident_id,
        incident_summary="Elevated checkout latency.",
        affected_service="checkout",
        investigation_status=InvestigationStatus.PARTIAL_SUCCESS,
        report_status=status,
        limitations=["Current evidence did not establish all incident details."],
        warnings=["Some source data was unavailable."],
        generated_at=datetime(2026, 11, 17, tzinfo=UTC),
    )


def _incident_payload(incident_id="INC-API"):
    return {
        "incident_id": incident_id,
        "description": "Elevated checkout latency.",
        "affected_service": "checkout",
        "environment": "production",
        "severity": "high",
        "start_time": "2026-11-17T10:00:00Z",
        "end_time": "2026-11-17T10:30:00Z",
        "additional_context": {"region": "west"},
    }


def _decision():
    return InvestigationDecision(
        incident_id="INC-API",
        decision_status=DecisionStatus.NEEDS_MORE_EVIDENCE,
        evidence_sufficiency=EvidenceSufficiency.PARTIAL,
        evidence_summary=InvestigationEvidenceSummary(),
        historical_current_alignment=HistoricalCurrentAlignment.NO_CURRENT,
        confidence=0.5,
        decision_rationale="More current evidence is needed.",
        recommended_next_stage=RecommendedNextStage.ADDITIONAL_EVIDENCE_COLLECTION,
    )


def test_valid_request_returns_complete_typed_partial_report(monkeypatch):
    calls = []

    async def fake_workflow(incident, *, llm, supervisor_llm, rca_llm):
        calls.append(incident)
        report = _report(incident_id=incident.incident_id)
        return {
            "investigation_report": report,
            "final_report": report,
            "supervisor_decision": _decision(),
        }

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    response = _client().post(
        "/api/v1/investigations",
        json=_incident_payload(),
        headers={"X-Correlation-ID": "test-correlation-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert str(UUID(body["investigation_id"])) == body["investigation_id"]
    assert body["investigation_id"] != "test-correlation-1"
    assert body["status"] == "partial"
    assert body["incident"]["incident_id"] == "INC-API"
    assert body["report"]["incident_summary"] == "Elevated checkout latency."
    assert body["supervisor_decision"]["decision_status"] == "needs_more_evidence"
    assert body["root_cause_analysis"] is None
    assert body["recommendations"] is None
    assert body["limitations"] == [
        "Current evidence did not establish all incident details."
    ]
    assert response.headers["X-Correlation-ID"] == "test-correlation-1"
    assert len(calls) == 1
    assert isinstance(calls[0], IncidentRequest)
    from app.schemas.api import InvestigationResponse

    InvestigationResponse.model_validate(body)


def test_invalid_request_returns_safe_422_without_echoing_input(monkeypatch):
    workflow_calls = []

    async def unexpected_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        workflow_calls.append(True)
        raise AssertionError("Invalid input must not run the graph.")

    monkeypatch.setattr(workflow, "run_investigation", unexpected_workflow)
    response = _client().post(
        "/api/v1/investigations",
        json={"description": "   ", "additional_context": {"private": "do-not-echo"}},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert "do-not-echo" not in response.text
    assert workflow_calls == []


def test_missing_report_returns_controlled_error(monkeypatch):
    async def missing_report(_incident, *, llm, supervisor_llm, rca_llm):
        return {"investigation_report": None, "final_report": None}

    monkeypatch.setattr(workflow, "run_investigation", missing_report)
    response = _client().post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )

    assert response.status_code == 500
    assert response.json()["error"] == "report_unavailable"
    assert "correlation_id" in response.json()


def test_workflow_exception_does_not_expose_secrets(monkeypatch):
    async def fail_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        raise RuntimeError("API_KEY=never-return-this")

    monkeypatch.setattr(workflow, "run_investigation", fail_workflow)
    response = _client().post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )

    assert response.status_code == 500
    assert response.json()["error"] == "workflow_execution_failed"
    assert "never-return-this" not in response.text
    assert "RuntimeError" not in response.text


def test_invalid_report_data_returns_controlled_error(monkeypatch):
    async def invalid_report(_incident, *, llm, supervisor_llm, rca_llm):
        return {
            "investigation_report": {
                "incident_summary": "Broken report",
                "investigation_status": "not-a-status",
            }
        }

    monkeypatch.setattr(workflow, "run_investigation", invalid_report)
    response = _client().post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )

    assert response.status_code == 500
    assert response.json()["error"] == "invalid_workflow_result"
    assert "not-a-status" not in response.text


def test_independent_requests_invoke_workflow_once_each_without_state_leak(monkeypatch):
    seen = []

    async def fake_workflow(incident, *, llm, supervisor_llm, rca_llm):
        seen.append(incident.incident_id)
        report = _report(incident_id=incident.incident_id)
        return {"investigation_report": report, "final_report": report}

    monkeypatch.setattr(workflow, "run_investigation", fake_workflow)
    client = _client()
    first = client.post(
        "/api/v1/investigations",
        json=_incident_payload("INC-FIRST"),
    )
    second = client.post(
        "/api/v1/investigations",
        json=_incident_payload("INC-SECOND"),
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["report"]["incident_id"] == "INC-FIRST"
    assert second.json()["report"]["incident_id"] == "INC-SECOND"
    assert seen == ["INC-FIRST", "INC-SECOND"]


def test_post_persists_history_report_evidence_agents_and_stats(monkeypatch):
    async def workflow_with_saved_data(
        incident,
        *,
        llm,
        supervisor_llm,
        rca_llm,
    ):
        report = _report(incident_id=incident.incident_id)
        return {
            "investigation_report": report,
            "final_report": report,
            "supervisor_decision": _decision(),
            "log_evidence": [
                {
                    "evidence_id": "log:evidence-1",
                    "source_type": "application_logs",
                    "source_name": "project logs",
                    "description": "Connection acquisition timed out.",
                    "classification": "current",
                }
            ],
            "execution_metadata": {
                "log_analysis:2026-11-17T10:01:00Z": {
                    "agent_name": "log_analysis",
                    "status": "success",
                    "duration_seconds": 0.25,
                    "evidence_count": 1,
                    "completed_at": "2026-11-17T10:01:01Z",
                }
            },
        }

    monkeypatch.setattr(workflow, "run_investigation", workflow_with_saved_data)
    client = _client()
    created = client.post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )
    assert created.status_code == 200
    investigation_id = created.json()["investigation_id"]

    history = client.get("/api/v1/investigations", params={"page": 1, "page_size": 5})
    matching_history = client.get(
        "/api/v1/investigations",
        params={"search": "INC-API"},
    )
    empty_search = client.get(
        "/api/v1/investigations",
        params={"search": "does-not-match"},
    )
    detail = client.get(f"/api/v1/investigations/{investigation_id}")
    report = client.get(f"/api/v1/investigations/{investigation_id}/report")
    evidence = client.get(f"/api/v1/investigations/{investigation_id}/evidence")
    agents = client.get(f"/api/v1/investigations/{investigation_id}/agents")
    stats = client.get("/api/v1/investigations/stats")

    assert (
        history.status_code
        == matching_history.status_code
        == empty_search.status_code
        == detail.status_code
        == report.status_code
        == 200
    )
    assert evidence.status_code == agents.status_code == stats.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["investigation_id"] == investigation_id
    assert matching_history.json()["total"] == 1
    assert matching_history.json()["items"][0]["investigation_id"] == investigation_id
    assert empty_search.json()["total"] == 0
    assert detail.json()["report"]["incident_summary"] == "Elevated checkout latency."
    assert report.json()["report"]["incident_summary"] == "Elevated checkout latency."
    assert evidence.json()["items"][0]["evidence_id"] == "log:evidence-1"
    assert agents.json()["agents"][0]["duration_ms"] == 250
    assert stats.json()["total_investigations"] == 1
    assert stats.json()["by_status"] == {"partial": 1}
    assert detail.json()["completed_at"] is not None


def test_history_archive_restore_and_permanent_delete(monkeypatch):
    async def workflow_with_report(incident, *, llm, supervisor_llm, rca_llm):
        report = _report(incident_id=incident.incident_id)
        return {"investigation_report": report, "final_report": report}

    monkeypatch.setattr(workflow, "run_investigation", workflow_with_report)
    client = _client()
    created = client.post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )
    investigation_id = created.json()["investigation_id"]

    archived = client.post(f"/api/v1/investigations/{investigation_id}/archive")
    active_history = client.get("/api/v1/investigations")
    archived_history = client.get(
        "/api/v1/investigations",
        params={"include_archived": True},
    )
    stats = client.get("/api/v1/investigations/stats")

    assert archived.status_code == 204
    assert active_history.json()["total"] == 0
    assert archived_history.json()["total"] == 1
    assert archived_history.json()["items"][0]["archived_at"] is not None
    assert stats.json()["total_investigations"] == 0
    assert client.get(f"/api/v1/investigations/{investigation_id}").status_code == 200

    restored = client.post(f"/api/v1/investigations/{investigation_id}/restore")
    assert restored.status_code == 204
    assert client.get("/api/v1/investigations").json()["total"] == 1

    deleted = client.delete(f"/api/v1/investigations/{investigation_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/investigations/{investigation_id}").status_code == 404
    assert client.get("/api/v1/investigations").json()["total"] == 0
    assert client.delete(
        "/api/v1/investigations/00000000-0000-0000-0000-000000000001"
    ).status_code == 404


def test_invalid_or_missing_history_ids_are_safe():
    client = _client()

    invalid_id = client.get("/api/v1/investigations/not-a-uuid")
    missing_id = client.get(
        "/api/v1/investigations/00000000-0000-0000-0000-000000000001"
    )

    assert invalid_id.status_code == 422
    assert missing_id.status_code == 404
    assert "traceback" not in missing_id.text.casefold()


def test_workflow_failure_is_saved_as_failed(monkeypatch):
    async def fail_workflow(_incident, *, llm, supervisor_llm, rca_llm):
        raise RuntimeError("sensitive-password=never-return-this")

    monkeypatch.setattr(workflow, "run_investigation", fail_workflow)
    client = _client()
    response = client.post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )

    assert response.status_code == 500
    assert "never-return-this" not in response.text
    with _TEST_SESSION() as session:
        from sqlalchemy import select

        from app.database.models import InvestigationRecord

        failed_id = session.scalar(select(InvestigationRecord.id))
    assert failed_id is not None
    detail = client.get(f"/api/v1/investigations/{failed_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"
    assert detail.json()["report"] is None
    stats = client.get("/api/v1/investigations/stats")
    assert stats.json()["by_status"] == {"failed": 1}


def test_database_failure_does_not_expose_connection_details(monkeypatch):
    def fail_before_workflow(*_args, **_kwargs):
        raise OperationalError(
            "INSERT",
            {},
            RuntimeError("postgresql://db-user:secret-password@localhost/private"),
        )

    monkeypatch.setattr(repository, "create_running", fail_before_workflow)
    response = _client().post(
        "/api/v1/investigations",
        json=_incident_payload(),
    )

    assert response.status_code == 503
    assert response.json()["error"] == "persistence_unavailable"
    assert "secret-password" not in response.text
    assert "postgresql://" not in response.text
