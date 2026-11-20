"""Add reversible archiving to investigation history.

Revision ID: 0002_investigation_archive
Revises: 0001_investigation_history
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_investigation_archive"
down_revision: Union[str, None] = "0001_investigation_history"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "investigations",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_investigations_archived_at",
        "investigations",
        ["archived_at"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade is disabled to preserve investigation archive state."
    )
