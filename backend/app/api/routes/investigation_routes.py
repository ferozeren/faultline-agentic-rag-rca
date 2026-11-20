"""Incident investigation endpoint."""

from datetime import UTC, datetime
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_investigation_service
from app.database import repository
from app.database.session import get_database_session
from app.schemas.api import APIErrorResponse, InvestigationResponse
from app.schemas.incident import IncidentRequest
from app.schemas.history import (
    InvestigationAgentRunsResponse,
    InvestigationDetailResponse,
    InvestigationEvidenceResponse,
    InvestigationHistoryResponse,
    InvestigationReportResponse,
    InvestigationStatsResponse,
)
from app.services.investigation_service import (
    InvestigationPersistenceError,
    InvestigationService,
    InvestigationServiceError,
)

router = APIRouter(prefix="/investigations", tags=["investigations"])
_INVESTIGATION_STATUSES = {
    "running",
    "pending",
    "in_progress",
    "completed",
    "success",
    "partial",
    "partial_success",
    "insufficient_evidence",
    "no_evidence",
    "failed",
}


def _validate_time_range(
    start_time: datetime | None,
    end_time: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    if start_time and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=UTC)
    if end_time and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=UTC)
    if start_time and end_time and start_time > end_time:
        raise HTTPException(status_code=422, detail="Invalid date range.")
    return start_time, end_time


@router.post(
    "",
    response_model=InvestigationResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": APIErrorResponse,
            "description": "Investigation could not be completed.",
        }
    },
)
async def create_investigation(
    incident: IncidentRequest,
    service: Annotated[InvestigationService, Depends(get_investigation_service)],
    session: Annotated[Session, Depends(get_database_session)],
) -> InvestigationResponse:
    """Run the existing investigation workflow once and return its report."""
    investigation_id = uuid4()
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        repository.create_running(session, investigation_id, incident, started_at)
    except (SQLAlchemyError, ValueError):
        raise InvestigationPersistenceError(
            "persistence_unavailable",
            "Investigation history storage is unavailable.",
        ) from None
    try:
        result = await service.investigate(incident)
    except InvestigationServiceError as error:
        try:
            repository.mark_failed(
                session,
                investigation_id,
                completed_at=datetime.now(UTC),
                duration_seconds=perf_counter() - started_clock,
                error_code=error.code,
            )
        except SQLAlchemyError:
            session.rollback()
        raise
    completed_at = datetime.now(UTC)
    try:
        repository.save_completed(
            session,
            investigation_id,
            report=result.report,
            supervisor_decision=result.supervisor_decision,
            evidence=result.evidence,
            execution_metadata=result.execution_metadata,
            completed_at=completed_at,
            duration_seconds=perf_counter() - started_clock,
        )
    except (SQLAlchemyError, LookupError, ValueError):
        session.rollback()
        try:
            repository.mark_failed(
                session,
                investigation_id,
                completed_at=datetime.now(UTC),
                duration_seconds=perf_counter() - started_clock,
                error_code="persistence_failed",
            )
        except SQLAlchemyError:
            session.rollback()
        raise InvestigationPersistenceError(
            "persistence_failed",
            "The investigation completed but could not be saved.",
        ) from None
    report = result.report
    response_status = (
        report.report_status.value
        if report.report_status is not None
        else report.investigation_status.value
    )
    return InvestigationResponse(
        investigation_id=str(investigation_id),
        status=response_status,
        incident=incident,
        supervisor_decision=result.supervisor_decision,
        report=report,
        root_cause_analysis=report.root_cause_analysis,
        recommendations=report.recommendation_result,
        timeline=report.timeline,
        evidence_analysis=report.evidence_analysis,
        evidence_references=report.evidence_references,
        conclusion=report.conclusion,
        warnings=report.warnings,
        limitations=report.limitations,
    )


@router.get("", response_model=InvestigationHistoryResponse)
def list_investigations(
    session: Annotated[Session, Depends(get_database_session)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    include_archived: bool = Query(default=False),
    service_name: str | None = Query(default=None, min_length=1, max_length=255),
    severity: str | None = Query(default=None, min_length=1, max_length=64),
    status_filter: str | None = Query(default=None, alias="status", min_length=1, max_length=64),
    search: str | None = Query(default=None, min_length=1, max_length=255),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> InvestigationHistoryResponse:
    start_time, end_time = _validate_time_range(start_time, end_time)
    if service_name is not None and not service_name.strip():
        raise HTTPException(status_code=422, detail="Invalid service filter.")
    if severity is not None and not severity.strip():
        raise HTTPException(status_code=422, detail="Invalid severity filter.")
    if status_filter is not None and status_filter not in _INVESTIGATION_STATUSES:
        raise HTTPException(status_code=422, detail="Invalid status filter.")
    if search is not None and not search.strip():
        raise HTTPException(status_code=422, detail="Invalid search query.")
    items, total = repository.list_investigations(
        session,
        page=page,
        page_size=page_size,
        include_archived=include_archived,
        service_name=service_name,
        severity=severity,
        status=status_filter,
        search=search.strip() if search else None,
        start_time=start_time,
        end_time=end_time,
    )
    return InvestigationHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=InvestigationStatsResponse)
def investigation_stats(
    session: Annotated[Session, Depends(get_database_session)],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> InvestigationStatsResponse:
    start_time, end_time = _validate_time_range(start_time, end_time)
    return InvestigationStatsResponse.model_validate(
        repository.get_stats(session, start_time=start_time, end_time=end_time)
    )


def _record_or_404(session: Session, investigation_id: UUID):
    record = repository.get_investigation(session, investigation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return record


@router.get("/{investigation_id}", response_model=InvestigationDetailResponse)
def get_investigation(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> InvestigationDetailResponse:
    record = _record_or_404(session, investigation_id)
    report_json = record.report.report_json if record.report else None
    return InvestigationDetailResponse(
        investigation_id=str(record.id),
        status=record.status,
        incident=record.incident_json,
        supervisor_decision=record.supervisor_decision,
        report=report_json,
        root_cause_analysis=report_json.get("root_cause_analysis") if report_json else None,
        recommendations=report_json.get("recommendation_result") if report_json else None,
        timeline=report_json.get("timeline", []) if report_json else [],
        evidence_analysis=report_json.get("evidence_analysis") if report_json else None,
        evidence_references=report_json.get("evidence_references", []) if report_json else [],
        conclusion=report_json.get("conclusion") if report_json else None,
        warnings=report_json.get("warnings", []) if report_json else [],
        limitations=report_json.get("limitations", []) if report_json else [],
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_seconds=record.duration_seconds,
    )


@router.get("/{investigation_id}/report", response_model=InvestigationReportResponse)
def get_investigation_report(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> InvestigationReportResponse:
    record = _record_or_404(session, investigation_id)
    if record.report is None:
        raise HTTPException(status_code=404, detail="Investigation report not found.")
    return InvestigationReportResponse(
        investigation_id=record.id,
        report=record.report.report_json,
    )


@router.get("/{investigation_id}/evidence", response_model=InvestigationEvidenceResponse)
def get_investigation_evidence(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> InvestigationEvidenceResponse:
    _record_or_404(session, investigation_id)
    items, total = repository.get_evidence(
        session,
        investigation_id,
        offset=offset,
        limit=limit,
    )
    return InvestigationEvidenceResponse(
        items=items,
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{investigation_id}/agents", response_model=InvestigationAgentRunsResponse)
def get_investigation_agents(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> InvestigationAgentRunsResponse:
    _record_or_404(session, investigation_id)
    return InvestigationAgentRunsResponse(
        investigation_id=investigation_id,
        agents=repository.get_agent_runs(session, investigation_id),
    )


@router.post(
    "/{investigation_id}/archive",
    status_code=status.HTTP_204_NO_CONTENT,
)
def archive_investigation(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> Response:
    try:
        found = repository.archive_investigation(session, investigation_id)
    except SQLAlchemyError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Investigation history storage is unavailable.",
        ) from None
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{investigation_id}/restore",
    status_code=status.HTTP_204_NO_CONTENT,
)
def restore_investigation(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> Response:
    try:
        found = repository.restore_investigation(session, investigation_id)
    except SQLAlchemyError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Investigation history storage is unavailable.",
        ) from None
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/{investigation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_investigation(
    investigation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> Response:
    try:
        found = repository.delete_investigation(session, investigation_id)
    except SQLAlchemyError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Investigation history storage is unavailable.",
        ) from None
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
