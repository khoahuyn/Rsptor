from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import String, Text, TIMESTAMP, ForeignKey, Index
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base

if TYPE_CHECKING:
    from .knowledge_base import ChatSessionORM


class MessageRole(str, PyEnum):
    """Message role types"""
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageORM(Base):
    __tablename__ = "messages"
    
    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False, 
        index=True
    )
    
    # Message content
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole),
        nullable=False,
        index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Optional metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)  # retrieval info, model settings, etc.
    
    # Token usage tracking
    input_tokens: Mapped[Optional[int]] = mapped_column()
    output_tokens: Mapped[Optional[int]] = mapped_column()
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    chat_session: Mapped["ChatSessionORM"] = relationship(
        "ChatSessionORM",
        back_populates="messages",
        lazy="select"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("ix_messages_session_role", "session_id", "role"),
    )
