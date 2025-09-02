from typing import Optional
from pydantic import BaseModel, Field


class CreateKBRequest(BaseModel):
    """Request to create Knowledge Base"""
    tenant_id: str = Field(...)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    settings: Optional[dict] = Field(None)


class KBResponse(BaseModel):
    """Knowledge Base response"""
    kb_id: str
    tenant_id: str
    name: str
    description: Optional[str]
    status: str
    document_count: int
    chunk_count: int
    total_tokens: Optional[int]
    settings: Optional[dict]
    created_at: str
    updated_at: Optional[str]


class CreateChatSessionRequest(BaseModel):
    """Request to create chat session"""
    tenant_id: str = Field(...)
    kb_id: str = Field(...)
    name: str = Field(..., min_length=1, max_length=255)
    system_prompt: Optional[str] = Field(None)
    settings: Optional[dict] = Field(None)


class ChatSessionResponse(BaseModel):
    """Chat session response"""
    session_id: str
    tenant_id: str
    kb_id: str
    name: str
    system_prompt: Optional[str]
    settings: Optional[dict]
    message_count: int
    created_at: str
    last_active: Optional[str]




