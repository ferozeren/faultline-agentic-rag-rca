"""HTTP response contracts for persisted investigation history."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.incident import IncidentRequest
from app.schemas.investigation_report import (
    EvidenceAnalysis,
    InvestigationConclusion,
    InvestigationReport,
    ReportEvidenceReference,
    RootCauseAnalysisSection,
)
from app.schemas.rca import RCATimelineEvent
from app.schemas.recommendations import RecommendationResult
from app.schemas.supervisor import InvestigationDecision


class InvestigationHistoryItem(BaseModel):
    investigation_id: UUID
    incident_id: str | None = None
    title: str
    service_name: str | None = None
    severity: str | None = None
    status: str
    incident_description: str
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    report_available: bool
    archived_at: datetime | None = None


class InvestigationHistoryResponse(BaseModel):
    items: list[InvestigationHistoryItem]
    total: int
    page: int
    page_size: int


class InvestigationDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation_id: str
    status: str
    incident: IncidentRequest
    supervisor_decision: InvestigationDecision | None = None
    report: InvestigationReport | None = None
    root_cause_analysis: RootCauseAnalysisSection | None = None
    recommendations: RecommendationResult | None = None
    timeline: list[RCATimelineEvent] = Field(default_factory=list)
    evidence_analysis: EvidenceAnalysis | None = None
    evidence_references: list[ReportEvidenceReference] = Field(default_factory=list)
    conclusion: InvestigationConclusion | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None


class InvestigationReportResponse(BaseModel):
    investigation_id: UUID
    report: InvestigationReport


class InvestigationEvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    agent_name: str | None = None
    evidence: dict[str, Any]
    created_at: datetime


class InvestigationEvidenceResponse(BaseModel):
    items: list[InvestigationEvidenceItem]
    total: int
    offset: int
    limit: int


class InvestigationAgentRun(BaseModel):
    agent_name: str
    status: str
    duration_ms: int | None = None
    result: dict[str, Any]
    error_message: str | None = None
    created_at: datetime


class InvestigationAgentRunsResponse(BaseModel):
    investigation_id: UUID
    agents: list[InvestigationAgentRun]


class InvestigationStatsResponse(BaseModel):
    total_investigations: int = Field(ge=0)
    by_severity: dict[str, int]
    by_status: dict[str, int]
    by_service: dict[str, int]
    start_time: datetime | None = None
    end_time: datetime | None = None

    model_config = ConfigDict(extra="forbid")
