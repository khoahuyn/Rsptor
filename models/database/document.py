from __future__ import annotations
from typing import List, Optional
from datetime import datetime

from sqlalchemy import String, Text, Integer, TIMESTAMP, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class DocumentORM(Base):
    __tablename__ = "documents"
    
    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"),
        nullable=False, 
        index=True
    )
    
    filename: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[Optional[str]] = mapped_column(String)
    processing_stats: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        onupdate=func.now()
    )
    
    knowledge_base: Mapped["KnowledgeBaseORM"] = relationship(
        "KnowledgeBaseORM",
        back_populates="documents",
        lazy="select"
    )
    chunks: Mapped[List["ChunkORM"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select"
    )
    __table_args__ = (
        Index("ix_documents_tenant_kb", "tenant_id", "kb_id"),
        Index("ix_documents_tenant_kb_checksum", "tenant_id", "kb_id", "checksum"),
    )


class ChunkORM(Base):
    __tablename__ = "chunks"
    
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    document: Mapped["DocumentORM"] = relationship(
        back_populates="chunks",
        lazy="select"
    )
    
    __table_args__ = (
        Index("ix_chunks_doc_index", "doc_id", "chunk_index"),
        Index("ix_chunks_doc_chunk_unique", "doc_id", "chunk_index", unique=True),
    )




