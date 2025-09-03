from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class CreateAssistantRequest(BaseModel):
    """Request model for creating an AI assistant"""
    name: str = Field(..., min_length=1, max_length=100, description="Assistant name")
    description: Optional[str] = Field(None, max_length=500, description="Assistant description")
    
    # Support both frontend format (knowledge_bases array) and direct kb_id
    knowledge_bases: Optional[List[str]] = Field(None, description="Knowledge base IDs (frontend format)")
    kb_id: Optional[str] = Field(None, description="Single knowledge base ID")
    tenant_id: Optional[str] = Field(None, description="Tenant/User ID")
    
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt for AI behavior")
    model_settings: Optional[Dict[str, Any]] = Field(None, description="Model configuration")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    
    def get_kb_id(self) -> str:
        """Get knowledge base ID from either format"""
        if self.knowledge_bases and len(self.knowledge_bases) > 0:
            return self.knowledge_bases[0]
        if self.kb_id:
            return self.kb_id
        raise ValueError("Either knowledge_bases or kb_id must be provided")
    
    def get_tenant_id(self) -> str:
        """Get tenant ID with default fallback"""
        return self.tenant_id or "test_tenant"


class UpdateAssistantRequest(BaseModel):
    """Request model for updating an AI assistant"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Assistant name")
    description: Optional[str] = Field(None, max_length=500, description="Assistant description")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt for AI behavior")
    model_settings: Optional[Dict[str, Any]] = Field(None, description="Model configuration")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")


class AssistantResponse(BaseModel):
    """Response model for AI assistant"""
    assistant_id: str = Field(..., description="Unique assistant ID")
    tenant_id: str = Field(..., description="Tenant/User ID")
    kb_id: str = Field(..., description="Knowledge Base ID")
    name: str = Field(..., description="Assistant name")
    description: Optional[str] = Field(None, description="Assistant description")
    system_prompt: Optional[str] = Field(None, description="System prompt for AI behavior")
    model_settings: Optional[Dict[str, Any]] = Field(None, description="Model configuration")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: Optional[str] = Field(None, description="Last update timestamp (ISO format)")
    
    # Frontend compatibility - knowledge_bases as array
    knowledge_bases: List[str] = Field(..., description="Knowledge Base IDs (array for frontend)")
    
    # Optional KB info (when requested with details)  
    kb_name: Optional[str] = Field(None, description="Knowledge Base name")
    kb_status: Optional[str] = Field(None, description="Knowledge Base status")


class AssistantListResponse(BaseModel):
    """Response model for listing assistants"""
    assistants: List[AssistantResponse] = Field(..., description="List of assistants")
    total: int = Field(..., description="Total number of assistants")
    has_more: bool = Field(..., description="Whether there are more assistants")


class AssistantStatsResponse(BaseModel):
    """Response model for assistant statistics"""
    assistant_id: str = Field(..., description="Assistant ID")
    name: str = Field(..., description="Assistant name")
    kb_name: Optional[str] = Field(None, description="Knowledge Base name")
    total_sessions: int = Field(..., description="Total chat sessions")
    total_messages: int = Field(..., description="Total messages across all sessions")
    created_at: str = Field(..., description="Creation timestamp")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


# For compatibility with frontend expectations (wrapped in ApiResponse format)
class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    code: int = Field(0, description="Response code (0 = success)")
    data: Any = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Optional message")
