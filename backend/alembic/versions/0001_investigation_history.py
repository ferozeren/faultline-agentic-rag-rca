"""Create persistent investigation history tables.

Revision ID: 0001_investigation_history
Revises:
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_investigation_history"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "investigations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("incident_id", sa.String(length=255), nullable=True),
        sa.Column("incident_title", sa.String(length=500), nullable=False),
        sa.Column("service_name", sa.String(length=255), nullable=True),
        sa.Column("severity", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("incident_description", sa.String(), nullable=False),
        sa.Column("incident_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("supervisor_decision", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("final_summary", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_investigations"),
    )
    op.create_index("ix_investigations_created_at", "investigations", ["created_at"])
    op.create_index("ix_investigations_service_name", "investigations", ["service_name"])
    op.create_index("ix_investigations_severity", "investigations", ["severity"])
    op.create_index("ix_investigations_status", "investigations", ["status"])

    op.create_table(
        "investigation_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("report_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            ["investigations.id"],
            name="fk_investigation_reports_investigation_id_investigations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_investigation_reports"),
        sa.UniqueConstraint(
            "investigation_id",
            name="uq_investigation_reports_investigation_id",
        ),
    )
    op.create_index(
        "ix_investigation_reports_investigation_id",
        "investigation_reports",
        ["investigation_id"],
    )

    op.create_table(
        "investigation_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evidence_id", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("agent_name", sa.String(length=255), nullable=True),
        sa.Column("evidence_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            ["investigations.id"],
            name="fk_investigation_evidence_investigation_id_investigations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_investigation_evidence"),
        sa.UniqueConstraint(
            "investigation_id",
            "source_type",
            "evidence_id",
            name="uq_investigation_evidence_source_evidence",
        ),
    )
    op.create_index(
        "ix_investigation_evidence_investigation_id",
        "investigation_evidence",
        ["investigation_id"],
    )

    op.create_table(
        "investigation_agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_name", sa.String(length=255), nullable=False),
        sa.Column("execution_status", sa.String(length=64), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            ["investigations.id"],
            name="fk_investigation_agent_runs_investigation_id_investigations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_investigation_agent_runs"),
        sa.UniqueConstraint(
            "investigation_id",
            "agent_name",
            name="uq_investigation_agent_runs_name",
        ),
    )
    op.create_index(
        "ix_investigation_agent_runs_investigation_id",
        "investigation_agent_runs",
        ["investigation_id"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade is disabled to protect persisted investigation history."
    )
