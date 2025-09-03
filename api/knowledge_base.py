from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from database.repository_factory import get_repositories
from models.database.knowledge_base import KnowledgeBaseStatus
from models import CreateKBRequest, KBResponse, CreateChatSessionRequest, ChatSessionResponse

router = APIRouter(prefix="/v1/kb", tags=["knowledge-base"])



@router.post("/create", response_model=KBResponse)
async def create_knowledge_base(request: CreateKBRequest):
    """Create a new Knowledge Base"""
    try:
        async with get_repositories() as repos:
            # Create KB
            kb = await repos.kb_repo.create_kb(
                tenant_id=request.tenant_id,
                name=request.name,
                description=request.description,
                settings=request.settings
            )
            
            # Convert to response model
            return KBResponse(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                name=kb.name,
                description=kb.description,
                status=kb.status.value,
                document_count=kb.document_count,
                chunk_count=kb.chunk_count,
                total_tokens=kb.total_tokens,
                settings=kb.settings,
                created_at=kb.created_at.isoformat(),
                updated_at=kb.updated_at.isoformat() if kb.updated_at else None
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create KB: {str(e)}")


@router.get("/list", response_model=List[KBResponse])
async def list_knowledge_bases(
    tenant_id: str = Query(..., description="Tenant/User ID"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List Knowledge Bases for a tenant"""
    try:
        async with get_repositories() as repos:
            # Parse status filter
            status_filter = None
            if status:
                try:
                    status_filter = KnowledgeBaseStatus(status.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
            
            # Get KBs
            kbs = await repos.kb_repo.get_kbs_by_tenant(tenant_id, status_filter)
            
            # Convert to response models
            return [
                KBResponse(
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    name=kb.name,
                    description=kb.description,
                    status=kb.status.value,
                    document_count=kb.document_count,
                    chunk_count=kb.chunk_count,
                    total_tokens=kb.total_tokens,
                    settings=kb.settings,
                    created_at=kb.created_at.isoformat(),
                    updated_at=kb.updated_at.isoformat() if kb.updated_at else None
                )
                for kb in kbs
            ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list KBs: {str(e)}")


@router.get("/{kb_id}/stats")
async def get_kb_statistics(kb_id: str):
    """Get detailed statistics for a Knowledge Base"""
    try:
        async with get_repositories() as repos:
            stats = await repos.kb_repo.get_kb_statistics(kb_id)
            if not stats:
                raise HTTPException(status_code=404, detail="Knowledge Base not found")
            return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KB stats: {str(e)}")
    

@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """Delete a Knowledge Base (cascade deletes documents, chunks, etc.)"""
    try:
        async with get_repositories() as repos:
            deleted = await repos.kb_repo.delete_by_id(kb_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Knowledge Base not found")
            
            return {"message": "Knowledge Base deleted successfully", "kb_id": kb_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete KB: {str(e)}")


@router.post("/chat/create", response_model=ChatSessionResponse)
async def create_chat_session(request: CreateChatSessionRequest):
    """Create a new chat session linked to a KB"""
    try:
        async with get_repositories() as repos:
            # Verify KB exists
            kb = await repos.kb_repo.get_by_id(request.kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge Base not found")
            
            # Create chat session
            session = await repos.chat_repo.create_chat_session(
                tenant_id=request.tenant_id,
                kb_id=request.kb_id,
                name=request.name,
                system_prompt=request.system_prompt,
                settings=request.settings
            )
            
            # Convert to response model
            return ChatSessionResponse(
                session_id=session.session_id,
                tenant_id=session.tenant_id,
                kb_id=session.kb_id,
                name=session.name,
                system_prompt=session.system_prompt,
                settings=session.settings,
                message_count=session.message_count,
                created_at=session.created_at.isoformat(),
                last_active=session.last_active.isoformat() if session.last_active else None
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")


@router.get("/chat/list", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    tenant_id: str = Query(..., description="Tenant/User ID"),
    kb_id: Optional[str] = Query(None, description="Filter by Knowledge Base ID")
):
    """List chat sessions for a tenant"""
    try:
        async with get_repositories() as repos:
            sessions = await repos.chat_repo.get_sessions_by_tenant(tenant_id, kb_id)
            
            # Convert to response models
            return [
                ChatSessionResponse(
                    session_id=session.session_id,
                    tenant_id=session.tenant_id,
                    kb_id=session.kb_id,
                    name=session.name,
                    system_prompt=session.system_prompt,
                    settings=session.settings,
                    message_count=session.message_count,
                    created_at=session.created_at.isoformat(),
                    last_active=session.last_active.isoformat() if session.last_active else None
                )
                for session in sessions
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {str(e)}")


@router.get("/chat/{kb_id}/sessions", response_model=List[ChatSessionResponse])
async def get_kb_chat_sessions(kb_id: str):
    """Get all chat sessions for a specific Knowledge Base"""
    try:
        async with get_repositories() as repos:
            sessions = await repos.chat_repo.get_sessions_by_kb(kb_id)
            
            # Convert to response models
            return [
                ChatSessionResponse(
                    session_id=session.session_id,
                    tenant_id=session.tenant_id,
                    kb_id=session.kb_id,
                    name=session.name,
                    system_prompt=session.system_prompt,
                    settings=session.settings,
                    message_count=session.message_count,
                    created_at=session.created_at.isoformat(),
                    last_active=session.last_active.isoformat() if session.last_active else None
                )
                for session in sessions
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KB chat sessions: {str(e)}")


@router.get("/{kb_id}/documents", response_model=dict)
async def get_kb_documents(
    kb_id: str,
    tenant_id: str = Query(..., description="Tenant/User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """Get documents for a specific Knowledge Base (for frontend compatibility)"""
    try:
        async with get_repositories() as repos:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Get documents
            documents = await repos.document_repo.get_documents_by_tenant_kb(
                tenant_id=tenant_id,
                kb_id=kb_id,
                limit=page_size,
                offset=offset
            )
            
            # Transform to frontend format
            documents_data = []
            for doc in documents:
                # Get real chunk count from database
                chunks = await repos.chunk_repo.get_chunks_by_document(doc.doc_id, order_by_index=False)
                real_chunk_count = len(chunks)
                
                # Get original processing stats
                processing_stats = doc.processing_stats or {}
                original_chunk_count = processing_stats.get("chunk_count", 0)
                
                documents_data.append({
                    "doc_id": doc.doc_id,
                    "source": doc.filename,  # Map filename to source
                    "tags": [],  # No tags in current model
                    "extra_meta": {
                        **processing_stats,
                        "original_chunk_count": original_chunk_count,  
                        "total_chunk_count": real_chunk_count,         
                        "raptor_chunk_count": real_chunk_count - original_chunk_count  # RAPTOR summary chunks
                    },
                    "chunk_count": real_chunk_count,  
                    "checksum": doc.checksum or "",
                    "created_at": doc.created_at.isoformat()
                })
            
            # Get total count (simplified - in real implementation should be optimized)
            all_docs = await repos.document_repo.get_documents_by_tenant_kb(
                tenant_id=tenant_id,
                kb_id=kb_id,
                limit=1000000  # Large number to get all
            )
            total_count = len(all_docs)
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "documents": documents_data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total_count,
                    "pages": total_pages
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KB documents: {str(e)}")





