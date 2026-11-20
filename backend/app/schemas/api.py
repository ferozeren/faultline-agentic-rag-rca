"""Validated API response contracts."""

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


class InvestigationResponse(BaseModel):
    """Report response with the complete original report retained."""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    incident: IncidentRequest
    supervisor_decision: InvestigationDecision | None = None
    report: InvestigationReport
    root_cause_analysis: RootCauseAnalysisSection | None = None
    recommendations: RecommendationResult | None = None
    timeline: list[RCATimelineEvent] = Field(default_factory=list)
    evidence_analysis: EvidenceAnalysis | None = None
    evidence_references: list[ReportEvidenceReference] = Field(default_factory=list)
    conclusion: InvestigationConclusion | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class APIErrorResponse(BaseModel):
    """Safe error body that never includes exception or incident payloads."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    correlation_id: str
