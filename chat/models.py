from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AssistantChatRequest(BaseModel):
    """Request for chatting with an AI assistant"""
    query: str = Field(...)
    assistant_id: str = Field(...)
    session_id: Optional[str] = Field(None)
    
    # Optional overrides (if not provided, use assistant's settings)
    include_context: bool = Field(True)
    max_context_messages: int = Field(10)


class AssistantChatResponse(BaseModel):
    """Response from assistant chat"""
    answer: str = Field(...)
    session_id: str = Field(...)
    message_id: str = Field(...)
    
    # ✅ ADD: Context passages for citations (Raptor-service style)
    context_passages: Optional[List[Dict[str, Any]]] = Field(None)
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(None)


class ChatMessage(BaseModel):
    """Individual chat message"""
    message_id: str
    role: str  # user, assistant, system
    content: str
    created_at: str
    extra_metadata: Optional[Dict[str, Any]] = None


class ChatSessionDetail(BaseModel):
    """Detailed chat session with messages"""
    session_id: str
    assistant_id: Optional[str] = None
    kb_id: str
    name: str
    messages: List[ChatMessage] = []
    message_count: int = 0
    created_at: str
    last_active: Optional[str] = None


class CreateChatSessionRequest(BaseModel):
    """Request to create a new chat session"""
    assistant_id: str = Field(...)
    name: Optional[str] = Field(None)


class CreateChatSessionResponse(BaseModel):
    """Response when creating a chat session"""
    session_id: str
    assistant_id: str
    name: str
    created_at: str
