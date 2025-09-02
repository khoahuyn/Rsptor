from datetime import datetime
from typing import List, Optional
from enum import Enum as PyEnum

from sqlalchemy import TIMESTAMP, String, Integer, Text, ForeignKey
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class KnowledgeBaseStatus(str, PyEnum):
    """Knowledge base status"""
    active = "active"
    inactive = "inactive"
    processing = "processing"
    error = "error"


class KnowledgeBaseORM(Base):
    """Knowledge Base for organizing documents and trees"""
    __tablename__ = "knowledge_bases"
    
    # Primary identifiers
    kb_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    
    # KB metadata
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[KnowledgeBaseStatus] = mapped_column(
        SAEnum(KnowledgeBaseStatus, name="kb_status"), 
        default=KnowledgeBaseStatus.active,
        nullable=False
    )
    
    # Statistics
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Configuration  
    settings: Mapped[Optional[dict]] = mapped_column(JSONB)  # Chunking, embedding settings
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        onupdate=func.now()
    )
    
    # Relationships
    documents: Mapped[List["DocumentORM"]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="select"
    )
    # trees relationship removed - using RAGFlow approach
    chat_sessions: Mapped[List["ChatSessionORM"]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="select"
    )


class ChatSessionORM(Base):
    """Chat sessions linked to Knowledge Bases (như RAGFlow)"""
    __tablename__ = "chat_sessions"
    
    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    
    # Link to Knowledge Base
    kb_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"),
        nullable=False, 
        index=True
    )
    
    # Session metadata
    name: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    
    # Chat settings
    settings: Mapped[Optional[dict]] = mapped_column(JSONB)  # Model, temperature, etc.
    
    # Statistics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    last_active: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        onupdate=func.now()
    )
    
    # Relationships
    knowledge_base: Mapped["KnowledgeBaseORM"] = relationship(back_populates="chat_sessions")




