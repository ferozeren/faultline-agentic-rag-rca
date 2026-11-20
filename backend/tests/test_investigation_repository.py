"""Repository and JSON safety tests use isolated SQLite, never the configured DB."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import (
    InvestigationAgentRunRecord,
    InvestigationEvidenceRecord,
    InvestigationRecord,
    InvestigationReportRecord,
)
from app.database import repository
from app.database.serialization import sanitize_json
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import (
    InvestigationReport,
    InvestigationStatus,
    ReportStatus,
)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session
    engine.dispose()


def _report():
    return InvestigationReport(
        incident_summary="Observed elevated latency.",
        investigation_status=InvestigationStatus.PARTIAL_SUCCESS,
        report_status=ReportStatus.PARTIAL,
        generated_at=datetime(2026, 11, 19, tzinfo=UTC),
    )


def test_json_sanitization_removes_secret_fields_and_evaluation_labels():
    result = sanitize_json(
        {
            "api_key": "secret-value",
            "nested": {
                "password": "do-not-store",
                "scenario_label": "internal-label",
                "message": "Authorization: Bearer abc.def.ghi",
            },
        }
    )

    assert result == {
        "nested": {"message": "Authorization: Bearer [REDACTED]"}
    }


def test_repository_persists_report_evidence_agents_and_supports_filters(session):
    investigation_id = uuid4()
    incident = IncidentRequest(
        incident_id="INC-PG-TEST",
        description="Database connection latency increased; password=secret-value",
        affected_service="checkout",
        severity="SEV-2",
        additional_context={"api_key": "must-not-persist"},
    )
    started_at = datetime(2026, 11, 19, 10, tzinfo=UTC)
    evidence = Evidence(
        evidence_id="evidence:one",
        source_type=EvidenceSourceType.APPLICATION_LOGS,
        source_name="application logs",
        description="Connection acquisition timed out.",
        classification="current",
    )
    repository.create_running(session, investigation_id, incident, started_at)
    repository.save_completed(
        session,
        investigation_id,
        report=_report(),
        supervisor_decision=None,
        evidence=[evidence, evidence],
        execution_metadata={
            "log_analysis:1": {
                "agent_name": "log_analysis",
                "status": "success",
                "duration_seconds": 0.125,
                "evidence_count": 1,
                "completed_at": "2026-11-19T10:00:01+00:00",
            }
        },
        completed_at=datetime(2026, 11, 19, 10, 0, 2, tzinfo=UTC),
        duration_seconds=2.0,
    )

    items, total = repository.list_investigations(
        session,
        page=1,
        page_size=10,
        service_name="checkout",
        status="partial",
    )
    searched_items, searched_total = repository.list_investigations(
        session,
        page=1,
        page_size=10,
        search="INC-PG-TEST",
    )
    wildcard_items, wildcard_total = repository.list_investigations(
        session,
        page=1,
        page_size=10,
        search="%",
    )
    stored = repository.get_investigation(session, investigation_id)
    evidence_items, evidence_total = repository.get_evidence(
        session, investigation_id, offset=0, limit=10
    )
    agent_runs = repository.get_agent_runs(session, investigation_id)

    assert total == 1
    assert searched_total == 1
    assert searched_items[0]["investigation_id"] == str(investigation_id)
    assert wildcard_total == 0
    assert wildcard_items == []
    assert items[0]["investigation_id"] == str(investigation_id)
    assert stored is not None
    assert stored.report.report_json["incident_summary"] == "Observed elevated latency."
    assert "api_key" not in stored.incident_json["additional_context"]
    assert "must-not-persist" not in stored.incident_json["additional_context"].values()
    assert "secret-value" not in stored.incident_description
    assert evidence_total == 1
    assert evidence_items[0]["evidence"]["evidence_id"] == "evidence:one"
    assert len(agent_runs) == 1
    assert agent_runs[0]["duration_ms"] == 125


def test_repository_empty_history_and_stats(session):
    items, total = repository.list_investigations(
        session, page=1, page_size=20
    )
    stats = repository.get_stats(session)

    assert items == []
    assert total == 0
    assert stats == {
        "total_investigations": 0,
        "by_severity": {},
        "by_status": {},
        "by_service": {},
        "start_time": None,
        "end_time": None,
    }


def test_failed_investigation_status_is_persisted(session):
    investigation_id = uuid4()
    started_at = datetime(2026, 11, 19, tzinfo=UTC)
    repository.create_running(
        session,
        investigation_id,
        IncidentRequest(description="Current database errors."),
        started_at,
    )
    repository.mark_failed(
        session,
        investigation_id,
        completed_at=datetime(2026, 11, 19, 0, 0, 5, tzinfo=UTC),
        duration_seconds=5,
    )

    record = repository.get_investigation(session, investigation_id)
    assert record is not None
    assert record.status == "failed"
    assert record.completed_at is not None
    assert record.error_message == "Investigation failed (investigation_failed)."


def test_archive_restore_and_permanent_delete_cascade(session):
    investigation_id = uuid4()
    started_at = datetime(2026, 11, 19, tzinfo=UTC)
    evidence = Evidence(
        evidence_id="evidence:archive-test",
        source_type=EvidenceSourceType.APPLICATION_LOGS,
        source_name="application logs",
        description="A bounded error signal.",
        classification="current",
    )
    repository.create_running(
        session,
        investigation_id,
        IncidentRequest(description="A service returned HTTP 503."),
        started_at,
    )
    repository.save_completed(
        session,
        investigation_id,
        report=_report(),
        supervisor_decision=None,
        evidence=[evidence],
        execution_metadata={
            "supervisor:run": {
                "agent_name": "supervisor",
                "status": "success",
                "duration_seconds": 0.1,
            }
        },
        completed_at=started_at,
        duration_seconds=0.1,
    )

    assert repository.archive_investigation(session, investigation_id)
    active_items, active_total = repository.list_investigations(
        session, page=1, page_size=10
    )
    all_items, all_total = repository.list_investigations(
        session, page=1, page_size=10, include_archived=True
    )
    assert active_items == []
    assert active_total == 0
    assert all_total == 1
    assert all_items[0]["archived_at"] is not None
    assert repository.get_stats(session)["total_investigations"] == 0
    session.rollback()

    assert repository.restore_investigation(session, investigation_id)
    restored_items, restored_total = repository.list_investigations(
        session, page=1, page_size=10
    )
    assert restored_total == 1
    assert restored_items[0]["archived_at"] is None
    session.rollback()

    assert repository.archive_investigation(session, investigation_id)
    assert repository.delete_investigation(session, investigation_id)
    assert repository.get_investigation(session, investigation_id) is None
    for model in (
        InvestigationReportRecord,
        InvestigationEvidenceRecord,
        InvestigationAgentRunRecord,
    ):
        assert session.scalar(select(func.count()).select_from(model)) == 0
    session.rollback()
    assert not repository.delete_investigation(session, investigation_id)
