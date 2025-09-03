from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from datetime import datetime

from sqlalchemy import String, Text, TIMESTAMP, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBaseORM, ChatSessionORM


class AssistantORM(Base):
    __tablename__ = "assistants"
    
    assistant_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"),
        nullable=False, 
        index=True
    )
    
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    model_settings: Mapped[Optional[dict]] = mapped_column(JSONB)
    avatar_url: Mapped[Optional[str]] = mapped_column(String)
    
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
    knowledge_base: Mapped["KnowledgeBaseORM"] = relationship(
        "KnowledgeBaseORM",
        back_populates="assistants",
        lazy="select"
    )
    chat_sessions: Mapped[List["ChatSessionORM"]] = relationship(
        "ChatSessionORM",
        back_populates="assistant",
        cascade="all, delete-orphan",
        lazy="select"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_assistants_tenant_kb", "tenant_id", "kb_id"),
        Index("ix_assistants_tenant_id", "tenant_id"),
    )
