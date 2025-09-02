"""fix_embedding_owner_type_enum_add_summary_root

Revision ID: 2d2637883dd7
Revises: c7a4b728ec0f
Create Date: 2025-09-02 19:17:47.435166

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2d2637883dd7'
down_revision = 'c7a4b728ec0f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add 'summary' and 'root' values to embedding_owner_type enum
    op.execute("ALTER TYPE embedding_owner_type ADD VALUE 'summary'")
    op.execute("ALTER TYPE embedding_owner_type ADD VALUE 'root'")


def downgrade() -> None:
    # NOTE: PostgreSQL doesn't support removing enum values once added
    # To downgrade, you would need to recreate the entire enum type
    pass



