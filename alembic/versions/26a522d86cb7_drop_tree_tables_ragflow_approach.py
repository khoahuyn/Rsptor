"""drop_tree_tables_ragflow_approach

Drop RAPTOR tree tables and enums - migrating to RAGFlow approach
that uses only embeddings for similarity search without tree persistence.

Revision ID: 26a522d86cb7
Revises: 2b2033901f4b
Create Date: 2025-08-29 14:40:28.357961

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '26a522d86cb7'
down_revision = '2b2033901f4b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop tree tables - migrating to RAGFlow approach with embeddings only"""
    
    # Drop tables in correct order (respecting foreign keys)
    print("Dropping tree tables for RAGFlow approach...")
    
    # 1. Drop tree_node_chunks (depends on tree_nodes and chunks)
    op.drop_table('raptor_tree_node_chunks')
    
    # 2. Drop tree_edges (depends on tree_nodes)  
    op.drop_table('raptor_tree_edges')
    
    # 3. Drop tree_nodes (depends on trees)
    op.drop_table('raptor_tree_nodes')
    
    # 4. Drop trees (base table)
    op.drop_table('raptor_trees')
    
    # 5. Drop enum type
    op.execute("DROP TYPE IF EXISTS raptor_node_kind CASCADE")
    
    print("Tree tables dropped - now using RAGFlow embeddings approach")


def downgrade() -> None:
    """Recreate tree tables - restore tree-based approach"""
    
    print("Recreating tree tables...")
    
    # 1. Recreate enum type
    raptor_node_kind = sa.Enum('leaf', 'parent', 'root', name='raptor_node_kind')
    raptor_node_kind.create(op.get_bind())
    
    # 2. Recreate trees table
    op.create_table('raptor_trees',
        sa.Column('tree_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('kb_id', sa.String(), nullable=False),
        sa.Column('doc_id', sa.String(), nullable=True),
        sa.Column('total_levels', sa.Integer(), nullable=False),
        sa.Column('total_nodes', sa.Integer(), nullable=False),
        sa.Column('algorithm', sa.String(), nullable=False),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.doc_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['kb_id'], ['knowledge_bases.kb_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('tree_id')
    )
    op.create_index('ix_raptor_trees_kb_id', 'raptor_trees', ['kb_id'])
    op.create_index('ix_raptor_trees_tenant_id', 'raptor_trees', ['tenant_id'])
    op.create_index('ix_raptor_trees_doc_id', 'raptor_trees', ['doc_id'])
    
    # 3. Recreate tree_nodes table
    op.create_table('raptor_tree_nodes',
        sa.Column('node_id', sa.String(), nullable=False),
        sa.Column('tree_id', sa.String(), nullable=False),
        sa.Column('level', sa.Integer(), nullable=False),
        sa.Column('kind', raptor_node_kind, nullable=False),
        sa.Column('cluster_id', sa.String(), nullable=True),
        sa.Column('layer_order', sa.Integer(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('parent_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('leaf_count', sa.Integer(), nullable=True),
        sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['tree_id'], ['raptor_trees.tree_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('node_id')
    )
    op.create_index('ix_raptor_tree_nodes_kind', 'raptor_tree_nodes', ['kind'])
    op.create_index('ix_raptor_tree_nodes_level', 'raptor_tree_nodes', ['level'])
    op.create_index('ix_raptor_tree_nodes_tree_id', 'raptor_tree_nodes', ['tree_id'])
    op.create_index('ix_tree_nodes_tree_level', 'raptor_tree_nodes', ['tree_id', 'level'])
    
    # 4. Recreate tree_edges table
    op.create_table('raptor_tree_edges',
        sa.Column('parent_id', sa.String(), nullable=False),
        sa.Column('child_id', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['child_id'], ['raptor_tree_nodes.node_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_id'], ['raptor_tree_nodes.node_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('parent_id', 'child_id')
    )
    op.create_index('ix_tree_edges_child', 'raptor_tree_edges', ['child_id'])
    op.create_index('ix_tree_edges_parent', 'raptor_tree_edges', ['parent_id'])
    
    # 5. Recreate tree_node_chunks table
    op.create_table('raptor_tree_node_chunks',
        sa.Column('node_id', sa.String(), nullable=False),
        sa.Column('chunk_id', sa.String(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunks.chunk_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['node_id'], ['raptor_tree_nodes.node_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('node_id', 'chunk_id')
    )
    op.create_index('ix_tree_node_chunks_chunk', 'raptor_tree_node_chunks', ['chunk_id'])
    op.create_index('ix_tree_node_chunks_node', 'raptor_tree_node_chunks', ['node_id'])
    op.create_index('ix_tree_node_chunks_rank', 'raptor_tree_node_chunks', ['node_id', 'rank'])
    
    print("Tree tables recreated")



