"""SQLAlchemy models for durable investigation history."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.base import Base

JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class InvestigationRecord(Base):
    __tablename__ = "investigations"
    __table_args__ = (
        Index("ix_investigations_created_at", "created_at"),
        Index("ix_investigations_service_name", "service_name"),
        Index("ix_investigations_severity", "severity"),
        Index("ix_investigations_status", "status"),
        Index("ix_investigations_archived_at", "archived_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    incident_id: Mapped[str | None] = mapped_column(String(255))
    incident_title: Mapped[str] = mapped_column(String(500), nullable=False)
    service_name: Mapped[str | None] = mapped_column(String(255))
    severity: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    incident_description: Mapped[str] = mapped_column(String, nullable=False)
    incident_json: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT, nullable=False
    )
    supervisor_decision: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    final_summary: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(String(500))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    report: Mapped[InvestigationReportRecord | None] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
        uselist=False,
    )
    evidence: Mapped[list[InvestigationEvidenceRecord]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
    )
    agent_runs: Mapped[list[InvestigationAgentRunRecord]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
    )


class InvestigationReportRecord(Base):
    __tablename__ = "investigation_reports"
    __table_args__ = (
        UniqueConstraint("investigation_id", name="uq_investigation_reports_investigation_id"),
        Index("ix_investigation_reports_investigation_id", "investigation_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    investigation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    report_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    investigation: Mapped[InvestigationRecord] = relationship(back_populates="report")


class InvestigationEvidenceRecord(Base):
    __tablename__ = "investigation_evidence"
    __table_args__ = (
        UniqueConstraint(
            "investigation_id",
            "source_type",
            "evidence_id",
            name="uq_investigation_evidence_source_evidence",
        ),
        Index("ix_investigation_evidence_investigation_id", "investigation_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    investigation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    evidence_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_name: Mapped[str | None] = mapped_column(String(255))
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    investigation: Mapped[InvestigationRecord] = relationship(
        back_populates="evidence"
    )


class InvestigationAgentRunRecord(Base):
    __tablename__ = "investigation_agent_runs"
    __table_args__ = (
        UniqueConstraint(
            "investigation_id",
            "agent_name",
            name="uq_investigation_agent_runs_name",
        ),
        Index("ix_investigation_agent_runs_investigation_id", "investigation_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    investigation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    agent_name: Mapped[str] = mapped_column(String(255), nullable=False)
    execution_status: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    investigation: Mapped[InvestigationRecord] = relationship(
        back_populates="agent_runs"
    )
