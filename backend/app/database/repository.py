"""Transactional data access for persisted investigations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.database.models import (
    InvestigationAgentRunRecord,
    InvestigationEvidenceRecord,
    InvestigationRecord,
    InvestigationReportRecord,
)
from app.database.serialization import sanitize_json
from app.schemas.evidence import Evidence
from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import InvestigationReport
from app.schemas.supervisor import InvestigationDecision

REPORT_VERSION = "1"
_SOURCE_AGENTS = {
    "historical_rag": "historical_rag",
    "application_logs": "log_analysis",
    "service_metrics": "metrics_analysis",
    "prometheus": "prometheus_analysis",
    "grafana": "grafana_analysis",
}


def _safe_text(value: str | None) -> str | None:
    return sanitize_json(value) if value is not None else None


def create_running(
    session: Session,
    investigation_id: UUID,
    incident: IncidentRequest,
    started_at: datetime,
) -> None:
    incident_json = sanitize_json(incident.model_dump(mode="json"))
    title = (
        _safe_text(incident.incident_id)
        or _safe_text(incident.affected_service)
        or _safe_text(incident.description)
        or "Incident"
    )
    with session.begin():
        session.add(
            InvestigationRecord(
                id=investigation_id,
                incident_id=_safe_text(incident.incident_id),
                incident_title=title[:500],
                service_name=_safe_text(incident.affected_service),
                severity=_safe_text(incident.severity),
                status="running",
                incident_description=_safe_text(incident.description) or "",
                incident_json=incident_json,
                started_at=started_at,
                created_at=started_at,
                updated_at=started_at,
            )
        )


def mark_failed(
    session: Session,
    investigation_id: UUID,
    *,
    completed_at: datetime,
    duration_seconds: float,
    error_code: str = "investigation_failed",
) -> None:
    with session.begin():
        record = session.get(InvestigationRecord, investigation_id)
        if record is None:
            return
        record.status = "failed"
        safe_code = (
            error_code
            if error_code.replace("_", "").isalnum()
            else "investigation_failed"
        )
        record.error_message = f"Investigation failed ({safe_code})."
        record.completed_at = completed_at
        record.duration_seconds = duration_seconds
        record.updated_at = completed_at


def save_completed(
    session: Session,
    investigation_id: UUID,
    *,
    report: InvestigationReport,
    supervisor_decision: InvestigationDecision | None,
    evidence: list[Evidence],
    execution_metadata: dict[str, Any],
    completed_at: datetime,
    duration_seconds: float,
) -> None:
    report_json = sanitize_json(report.model_dump(mode="json"))
    decision_json = (
        sanitize_json(supervisor_decision.model_dump(mode="json"))
        if supervisor_decision
        else None
    )
    with session.begin():
        record = session.get(InvestigationRecord, investigation_id)
        if record is None:
            raise LookupError("Investigation record was not found.")
        record.status = (
            report.report_status.value
            if report.report_status is not None
            else report.investigation_status.value
        )
        record.error_message = None
        record.supervisor_decision = decision_json
        record.final_summary = _safe_text(_report_summary(report))
        record.completed_at = completed_at
        record.duration_seconds = duration_seconds
        record.updated_at = completed_at

        report_record = session.scalar(
            select(InvestigationReportRecord).where(
                InvestigationReportRecord.investigation_id == investigation_id
            )
        )
        if report_record is None:
            report_record = InvestigationReportRecord(
                id=uuid4(),
                investigation_id=investigation_id,
                report_json=report_json,
                report_version=REPORT_VERSION,
                created_at=completed_at,
            )
            session.add(report_record)
        else:
            report_record.report_json = report_json
            report_record.report_version = REPORT_VERSION

        session.query(InvestigationEvidenceRecord).filter_by(
            investigation_id=investigation_id
        ).delete(synchronize_session=False)
        unique_evidence = {(item.source_type.value, item.evidence_id): item for item in evidence}
        for item in unique_evidence.values():
            payload = sanitize_json(item.model_dump(mode="json"))
            session.add(
                InvestigationEvidenceRecord(
                    id=uuid4(),
                    investigation_id=investigation_id,
                    evidence_id=_safe_text(item.evidence_id),
                    source_type=_safe_text(item.source_type.value),
                    agent_name=_safe_text(_SOURCE_AGENTS.get(item.source_type.value)),
                    evidence_json=payload,
                    created_at=completed_at,
                )
            )

        session.query(InvestigationAgentRunRecord).filter_by(
            investigation_id=investigation_id
        ).delete(synchronize_session=False)
        for metadata in execution_metadata.values():
            if not isinstance(metadata, dict):
                continue
            agent_name = metadata.get("agent_name")
            status = metadata.get("status")
            if not isinstance(agent_name, str) or not isinstance(status, str):
                continue
            duration_seconds_value = metadata.get("duration_seconds")
            duration_ms = (
                round(duration_seconds_value * 1000)
                if isinstance(duration_seconds_value, (int, float))
                and not isinstance(duration_seconds_value, bool)
                else None
            )
            error_type = metadata.get("error_type")
            error_message = (
                f"{agent_name} failed ({error_type})."
                if status == "failed" and isinstance(error_type, str)
                else None
            )
            session.add(
                InvestigationAgentRunRecord(
                    id=uuid4(),
                    investigation_id=investigation_id,
                    agent_name=agent_name,
                    execution_status=status,
                    duration_ms=duration_ms,
                    result_json=sanitize_json(metadata),
                    error_message=error_message,
                    created_at=_metadata_timestamp(metadata) or completed_at,
                )
            )


def _report_summary(report: InvestigationReport) -> str:
    if report.executive_summary is not None:
        return report.executive_summary.summary.text
    return report.incident_summary


def _metadata_timestamp(metadata: dict[str, Any]) -> datetime | None:
    value = metadata.get("completed_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def list_investigations(
    session: Session,
    *,
    page: int,
    page_size: int,
    include_archived: bool = False,
    service_name: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    search: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> tuple[list[dict[str, Any]], int]:
    statement = select(InvestigationRecord)
    count_statement = select(func.count()).select_from(InvestigationRecord)
    filters = _filters(
        service_name=service_name,
        severity=severity,
        status=status,
        search=search,
        start_time=start_time,
        end_time=end_time,
        include_archived=include_archived,
    )
    if filters:
        statement = statement.where(*filters)
        count_statement = count_statement.where(*filters)
    total = session.scalar(count_statement) or 0
    records = session.scalars(
        statement.options(selectinload(InvestigationRecord.report)).order_by(
            InvestigationRecord.created_at.desc(),
            InvestigationRecord.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return [_list_item(record) for record in records], total


def _filters(
    *,
    service_name: str | None,
    severity: str | None,
    status: str | None,
    start_time: datetime | None,
    end_time: datetime | None,
    search: str | None = None,
    include_archived: bool = False,
) -> list[Any]:
    filters = []
    if not include_archived:
        filters.append(InvestigationRecord.archived_at.is_(None))
    if service_name:
        filters.append(InvestigationRecord.service_name == service_name)
    if severity:
        filters.append(InvestigationRecord.severity == severity)
    if status:
        filters.append(InvestigationRecord.status == status)
    if search:
        escaped = (
            search.strip()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        pattern = f"%{escaped}%"
        filters.append(
            or_(
                InvestigationRecord.incident_id.ilike(pattern, escape="\\"),
                InvestigationRecord.incident_title.ilike(pattern, escape="\\"),
                InvestigationRecord.incident_description.ilike(
                    pattern,
                    escape="\\",
                ),
                InvestigationRecord.service_name.ilike(pattern, escape="\\"),
            )
        )
    if start_time:
        filters.append(InvestigationRecord.created_at >= start_time)
    if end_time:
        filters.append(InvestigationRecord.created_at <= end_time)
    return filters


def _list_item(record: InvestigationRecord) -> dict[str, Any]:
    return {
        "investigation_id": str(record.id),
        "incident_id": record.incident_id,
        "title": record.incident_title,
        "service_name": record.service_name,
        "severity": record.severity,
        "status": record.status,
        "incident_description": record.incident_description,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "duration_seconds": record.duration_seconds,
        "report_available": record.report is not None,
        "archived_at": record.archived_at,
    }


def archive_investigation(session: Session, investigation_id: UUID) -> bool:
    with session.begin():
        record = session.get(InvestigationRecord, investigation_id)
        if record is None:
            return False
        if record.archived_at is None:
            archived_at = datetime.now(UTC)
            record.archived_at = archived_at
            record.updated_at = archived_at
    return True


def restore_investigation(session: Session, investigation_id: UUID) -> bool:
    with session.begin():
        record = session.get(InvestigationRecord, investigation_id)
        if record is None:
            return False
        if record.archived_at is not None:
            record.archived_at = None
            record.updated_at = datetime.now(UTC)
    return True


def delete_investigation(session: Session, investigation_id: UUID) -> bool:
    with session.begin():
        record = session.get(InvestigationRecord, investigation_id)
        if record is None:
            return False
        session.delete(record)
    return True


def get_investigation(
    session: Session,
    investigation_id: UUID,
) -> InvestigationRecord | None:
    return session.scalar(
        select(InvestigationRecord)
        .options(
            selectinload(InvestigationRecord.report),
            selectinload(InvestigationRecord.evidence),
            selectinload(InvestigationRecord.agent_runs),
        )
        .where(InvestigationRecord.id == investigation_id)
    )


def get_evidence(
    session: Session,
    investigation_id: UUID,
    *,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    count = session.scalar(
        select(func.count())
        .select_from(InvestigationEvidenceRecord)
        .where(InvestigationEvidenceRecord.investigation_id == investigation_id)
    ) or 0
    items = session.scalars(
        select(InvestigationEvidenceRecord)
        .where(InvestigationEvidenceRecord.investigation_id == investigation_id)
        .order_by(InvestigationEvidenceRecord.created_at, InvestigationEvidenceRecord.id)
        .offset(offset)
        .limit(limit)
    ).all()
    return [
        {
            "evidence_id": item.evidence_id,
            "source_type": item.source_type,
            "agent_name": item.agent_name,
            "evidence": item.evidence_json,
            "created_at": item.created_at,
        }
        for item in items
    ], count


def get_agent_runs(
    session: Session,
    investigation_id: UUID,
) -> list[dict[str, Any]]:
    items = session.scalars(
        select(InvestigationAgentRunRecord)
        .where(InvestigationAgentRunRecord.investigation_id == investigation_id)
        .order_by(InvestigationAgentRunRecord.created_at, InvestigationAgentRunRecord.id)
    ).all()
    return [
        {
            "agent_name": item.agent_name,
            "status": item.execution_status,
            "duration_ms": item.duration_ms,
            "result": item.result_json,
            "error_message": item.error_message,
            "created_at": item.created_at,
        }
        for item in items
    ]


def get_stats(
    session: Session,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, Any]:
    filters = _filters(
        service_name=None,
        severity=None,
        status=None,
        search=None,
        start_time=start_time,
        end_time=end_time,
    )
    total_statement = select(func.count()).select_from(InvestigationRecord)
    if filters:
        total_statement = total_statement.where(*filters)
    total = session.scalar(total_statement) or 0

    def grouped(column):
        statement = select(column, func.count()).select_from(InvestigationRecord)
        if filters:
            statement = statement.where(*filters)
        rows = session.execute(statement.group_by(column)).all()
        return {key: count for key, count in rows if key is not None}

    return {
        "total_investigations": total,
        "by_severity": grouped(InvestigationRecord.severity),
        "by_status": grouped(InvestigationRecord.status),
        "by_service": grouped(InvestigationRecord.service_name),
        "start_time": start_time,
        "end_time": end_time,
    }
