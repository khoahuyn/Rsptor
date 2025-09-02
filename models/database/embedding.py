from __future__ import annotations
from enum import Enum as PyEnum
from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Integer, TIMESTAMP, Index
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from .base import Base


class EmbeddingOwnerType(str, PyEnum):
    chunk = "chunk"
    summary = "summary"
    root = "root"


class EmbeddingORM(Base):
    __tablename__ = "embeddings"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    kb_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    
    owner_type: Mapped[EmbeddingOwnerType] = mapped_column(
        SAEnum(EmbeddingOwnerType, name="embedding_owner_type"),
        nullable=False,
        index=True
    )
    owner_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    
    model: Mapped[str] = mapped_column(String, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[List[float]] = mapped_column(Vector(), nullable=False)
    
    meta: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    __table_args__ = (
        Index("ix_embeddings_tenant_kb_owner", "tenant_id", "kb_id", "owner_type", "owner_id"),
        Index(
            "ix_embeddings_vector_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"}
        ),
        Index(
            "ix_embeddings_vector_ivfflat", 
            "vector",
            postgresql_using="ivfflat",
            postgresql_ops={"vector": "vector_cosine_ops"},
            postgresql_with={"lists": 100}
        ),
    )




