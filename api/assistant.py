from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path

from database.repository_factory import get_repositories
from models import (
    CreateAssistantRequest, UpdateAssistantRequest, AssistantResponse,
    AssistantListResponse, AssistantStatsResponse, ApiResponse
)

router = APIRouter(prefix="/v1/ai/assistants", tags=["AI Assistants"])


@router.post("", response_model=ApiResponse)
async def create_assistant(request: CreateAssistantRequest):
    """Create a new AI assistant"""
    try:
        async with get_repositories() as repos:
            # Extract tenant_id and kb_id from either format
            tenant_id = request.get_tenant_id()
            kb_id = request.get_kb_id()
            
            # Verify KB exists
            kb = await repos.kb_repo.get_by_id(kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge Base not found")
            
            # Create assistant
            assistant = await repos.assistant_repo.create_assistant(
                tenant_id=tenant_id,
                kb_id=kb_id,
                name=request.name,
                description=request.description,
                system_prompt=request.system_prompt,
                model_settings=request.model_settings,
                avatar_url=request.avatar_url
            )
            
            # Convert to response model
            response_data = AssistantResponse(
                assistant_id=assistant.assistant_id,
                tenant_id=assistant.tenant_id,
                kb_id=assistant.kb_id,
                name=assistant.name,
                description=assistant.description,
                system_prompt=assistant.system_prompt,
                model_settings=assistant.model_settings,
                avatar_url=assistant.avatar_url,
                created_at=assistant.created_at.isoformat(),
                updated_at=assistant.updated_at.isoformat() if assistant.updated_at else None,
                knowledge_bases=[assistant.kb_id],  # Convert single kb_id to array
                kb_name=kb.name,
                kb_status=kb.status.value
            )
            
            return ApiResponse(code=0, data=response_data)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")


@router.get("", response_model=AssistantListResponse)
async def list_assistants(
    user_id: Optional[str] = Query(None, description="User/Tenant ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of assistants to return"),
    offset: int = Query(0, ge=0, description="Number of assistants to skip")
):
    """List AI assistants for a user"""
    try:
        # Use default tenant if user_id not provided (for development)
        tenant_id = user_id or "test_tenant"
        
        async with get_repositories() as repos:
            assistants = await repos.assistant_repo.get_assistants_by_tenant(
                tenant_id=tenant_id,
                limit=limit,
                offset=offset
            )
            
            # Check if there are more assistants
            next_page_assistants = await repos.assistant_repo.get_assistants_by_tenant(
                tenant_id=tenant_id,
                limit=1,
                offset=offset + limit
            )
            has_more = len(next_page_assistants) > 0
            
            # Convert to response models
            assistant_responses = []
            for assistant in assistants:
                # Load KB info
                kb = await repos.kb_repo.get_by_id(assistant.kb_id)
                
                assistant_responses.append(AssistantResponse(
                    assistant_id=assistant.assistant_id,
                    tenant_id=assistant.tenant_id,
                    kb_id=assistant.kb_id,
                    name=assistant.name,
                    description=assistant.description,
                    system_prompt=assistant.system_prompt,
                    model_settings=assistant.model_settings,
                    avatar_url=assistant.avatar_url,
                    created_at=assistant.created_at.isoformat(),
                    updated_at=assistant.updated_at.isoformat() if assistant.updated_at else None,
                    knowledge_bases=[assistant.kb_id],  # Convert single kb_id to array
                    kb_name=kb.name if kb else None,
                    kb_status=kb.status.value if kb else None
                ))
            
            return AssistantListResponse(
                assistants=assistant_responses,
                total=len(assistant_responses),
                has_more=has_more
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list assistants: {str(e)}")


@router.get("/{assistant_id}", response_model=ApiResponse)
async def get_assistant(assistant_id: str = Path(..., description="Assistant ID")):
    """Get a specific assistant by ID"""
    try:
        async with get_repositories() as repos:
            assistant = await repos.assistant_repo.get_assistant_with_kb(assistant_id)
            if not assistant:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            # Convert to response model
            response_data = AssistantResponse(
                assistant_id=assistant.assistant_id,
                tenant_id=assistant.tenant_id,
                kb_id=assistant.kb_id,
                name=assistant.name,
                description=assistant.description,
                system_prompt=assistant.system_prompt,
                model_settings=assistant.model_settings,
                avatar_url=assistant.avatar_url,
                created_at=assistant.created_at.isoformat(),
                updated_at=assistant.updated_at.isoformat() if assistant.updated_at else None,
                knowledge_bases=[assistant.kb_id],  # Convert single kb_id to array
                kb_name=assistant.knowledge_base.name if assistant.knowledge_base else None,
                kb_status=assistant.knowledge_base.status.value if assistant.knowledge_base else None
            )
            
            return ApiResponse(code=0, data=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assistant: {str(e)}")


@router.get("/{assistant_id}/details", response_model=ApiResponse)
async def get_assistant_with_datasets(assistant_id: str = Path(..., description="Assistant ID")):
    """Get assistant with full knowledge base details"""
    try:
        async with get_repositories() as repos:
            assistant = await repos.assistant_repo.get_assistant_with_kb(assistant_id)
            if not assistant:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            # Convert to response model with extended KB info
            response_data = AssistantResponse(
                assistant_id=assistant.assistant_id,
                tenant_id=assistant.tenant_id,
                kb_id=assistant.kb_id,
                name=assistant.name,
                description=assistant.description,
                system_prompt=assistant.system_prompt,
                model_settings=assistant.model_settings,
                avatar_url=assistant.avatar_url,
                created_at=assistant.created_at.isoformat(),
                updated_at=assistant.updated_at.isoformat() if assistant.updated_at else None,
                knowledge_bases=[assistant.kb_id],  # Convert single kb_id to array
                kb_name=assistant.knowledge_base.name if assistant.knowledge_base else None,
                kb_status=assistant.knowledge_base.status.value if assistant.knowledge_base else None
            )
            
            return ApiResponse(code=0, data=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assistant details: {str(e)}")


@router.put("/{assistant_id}", response_model=ApiResponse)
async def update_assistant(
    assistant_id: str = Path(..., description="Assistant ID"),
    request: UpdateAssistantRequest = ...
):
    """Update an existing assistant"""
    try:
        async with get_repositories() as repos:
            # Check if assistant exists
            existing_assistant = await repos.assistant_repo.get_by_id(assistant_id)
            if not existing_assistant:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            # Prepare update data (only include non-None fields)
            update_data = {}
            if request.name is not None:
                update_data["name"] = request.name
            if request.description is not None:
                update_data["description"] = request.description
            if request.system_prompt is not None:
                update_data["system_prompt"] = request.system_prompt
            if request.model_settings is not None:
                update_data["model_settings"] = request.model_settings
            if request.avatar_url is not None:
                update_data["avatar_url"] = request.avatar_url
            
            # Update assistant
            updated_assistant = await repos.assistant_repo.update_assistant(
                assistant_id, **update_data
            )
            
            if not updated_assistant:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            # Load KB info
            kb = await repos.kb_repo.get_by_id(updated_assistant.kb_id)
            
            # Convert to response model
            response_data = AssistantResponse(
                assistant_id=updated_assistant.assistant_id,
                tenant_id=updated_assistant.tenant_id,
                kb_id=updated_assistant.kb_id,
                name=updated_assistant.name,
                description=updated_assistant.description,
                system_prompt=updated_assistant.system_prompt,
                model_settings=updated_assistant.model_settings,
                avatar_url=updated_assistant.avatar_url,
                created_at=updated_assistant.created_at.isoformat(),
                updated_at=updated_assistant.updated_at.isoformat() if updated_assistant.updated_at else None,
                knowledge_bases=[updated_assistant.kb_id],  # Convert single kb_id to array
                kb_name=kb.name if kb else None,
                kb_status=kb.status.value if kb else None
            )
            
            return ApiResponse(code=0, data=response_data)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update assistant: {str(e)}")


@router.delete("/{assistant_id}", response_model=ApiResponse)
async def delete_assistant(assistant_id: str = Path(..., description="Assistant ID")):
    """Delete an assistant"""
    try:
        async with get_repositories() as repos:
            deleted = await repos.assistant_repo.delete_assistant(assistant_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            return ApiResponse(code=0, data={"message": "Assistant deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete assistant: {str(e)}")


@router.get("/{assistant_id}/stats", response_model=ApiResponse)
async def get_assistant_stats(assistant_id: str = Path(..., description="Assistant ID")):
    """Get assistant usage statistics"""
    try:
        async with get_repositories() as repos:
            stats = await repos.assistant_repo.get_assistant_stats(assistant_id)
            if not stats:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            response_data = AssistantStatsResponse(**stats)
            return ApiResponse(code=0, data=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assistant stats: {str(e)}")
