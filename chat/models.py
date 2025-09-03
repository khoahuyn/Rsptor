from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SmartChatRequest(BaseModel):
    query: str
    tenant_id: str
    kb_id: str


class SmartChatResponse(BaseModel):
    answer: str


class AssistantChatRequest(BaseModel):
    """Request for chatting with an AI assistant"""
    query: str = Field(..., description="User query/message")
    assistant_id: str = Field(..., description="Assistant ID to chat with")
    session_id: Optional[str] = Field(None, description="Chat session ID (optional, will create if not provided)")
    
    # Optional overrides (if not provided, use assistant's settings)
    include_context: bool = Field(True, description="Whether to include conversation history")
    max_context_messages: int = Field(10, description="Maximum previous messages to include in context")


class AssistantChatResponse(BaseModel):
    """Response from assistant chat"""
    answer: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID of the response")
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (tokens, retrieval info, etc.)")


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
    assistant_id: str = Field(..., description="Assistant ID")
    name: Optional[str] = Field(None, description="Session name (auto-generated if not provided)")


class CreateChatSessionResponse(BaseModel):
    """Response when creating a chat session"""
    session_id: str
    assistant_id: str
    name: str
    created_at: str
