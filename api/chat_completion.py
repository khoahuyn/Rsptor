from fastapi import APIRouter, HTTPException, Depends, Path
from typing import List

from chat.models import (
    AssistantChatRequest, AssistantChatResponse,
    CreateChatSessionRequest, CreateChatSessionResponse, ChatSessionDetail, ChatMessage
)
from chat.service import get_chat_service, ChatService
from database.repository_factory import get_repositories


router = APIRouter(prefix="/v1/chat", tags=["Assistant Chat"])



@router.post("/assistant", response_model=AssistantChatResponse)
async def assistant_chat(
    request: AssistantChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> AssistantChatResponse:
    """Chat with an AI assistant, includes conversation history and session management"""
    try:
        return await chat_service.assistant_chat(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=CreateChatSessionResponse)
async def create_chat_session(request: CreateChatSessionRequest):
    """Create a new chat session for an assistant"""
    try:
        async with get_repositories() as repos:
            # Verify assistant exists
            assistant = await repos.assistant_repo.get_by_id(request.assistant_id)
            if not assistant:
                raise HTTPException(status_code=404, detail="Assistant not found")
            
            # Generate session name if not provided
            session_name = request.name or f"Chat with {assistant.name}"
            
            # Create session
            session = await repos.chat_repo.create_chat_session(
                tenant_id=assistant.tenant_id,
                kb_id=assistant.kb_id,
                name=session_name,
                system_prompt=assistant.system_prompt,
                settings=assistant.model_settings
            )
            
            # Update assistant_id link
            await repos.chat_repo.update_by_id(session.session_id, assistant_id=request.assistant_id)
            
            return CreateChatSessionResponse(
                session_id=session.session_id,
                assistant_id=request.assistant_id,
                name=session.name,
                created_at=session.created_at.isoformat()
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_chat_session(session_id: str = Path(..., description="Chat session ID")):
    """Get chat session details with message history"""
    try:
        async with get_repositories() as repos:
            # Get session
            session = await repos.chat_repo.get_by_id(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            # Get messages
            messages = await repos.message_repo.get_session_messages(session_id, limit=100)
            
            # Convert to response format
            chat_messages = [
                ChatMessage(
                    message_id=msg.message_id,
                    role=msg.role.value,
                    content=msg.content,
                    created_at=msg.created_at.isoformat(),
                    extra_metadata=msg.extra_metadata
                )
                for msg in messages
            ]
            
            return ChatSessionDetail(
                session_id=session.session_id,
                assistant_id=session.assistant_id,
                kb_id=session.kb_id,
                name=session.name,
                messages=chat_messages,
                message_count=len(chat_messages),
                created_at=session.created_at.isoformat(),
                last_active=session.last_active.isoformat() if session.last_active else None
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat session: {str(e)}")


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str = Path(..., description="Chat session ID"),
    limit: int = 50,
    offset: int = 0
):
    """Get messages for a chat session with pagination"""
    try:
        async with get_repositories() as repos:
            # Verify session exists
            session = await repos.chat_repo.get_by_id(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            # Get messages
            messages = await repos.message_repo.get_session_messages(
                session_id, limit=limit, offset=offset
            )
            
            return [
                ChatMessage(
                    message_id=msg.message_id,
                    role=msg.role.value,
                    content=msg.content,
                    created_at=msg.created_at.isoformat(),
                    extra_metadata=msg.extra_metadata
                )
                for msg in messages
            ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session messages: {str(e)}")


@router.get("/assistants/{assistant_id}/sessions")
async def get_assistant_sessions(assistant_id: str = Path(..., description="Assistant ID")):
    """Get all chat sessions for an assistant"""
    try:
        async with get_repositories() as repos:
            # Get all sessions for this assistant
            # Use direct SQL query since we need to filter by assistant_id
            from sqlalchemy import select
            from models.database.knowledge_base import ChatSessionORM
            
            stmt = select(ChatSessionORM).where(ChatSessionORM.assistant_id == assistant_id).order_by(ChatSessionORM.created_at.desc())
            result = await repos.chat_repo.session.execute(stmt)
            sessions = list(result.scalars().all())
            
            return [
                {
                    "session_id": session.session_id,
                    "name": session.name,
                    "message_count": session.message_count or 0,
                    "created_at": session.created_at.isoformat(),
                    "last_active": session.last_active.isoformat() if session.last_active else None
                }
                for session in sessions
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assistant sessions: {str(e)}")


@router.post("/sessions/{session_id}/sync-count")
async def sync_session_message_count(session_id: str = Path(..., description="Chat session ID")):
    """Sync session message count with actual messages in database"""
    try:
        async with get_repositories() as repos:
            session = await repos.chat_repo.sync_message_count(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            return {
                "session_id": session.session_id,
                "synced_message_count": session.message_count,
                "message": "Message count synced successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync message count: {str(e)}")


@router.post("/sessions/sync-all-counts")
async def sync_all_session_message_counts():
    """Sync message counts for all sessions"""
    try:
        async with get_repositories() as repos:
            # Get all sessions
            from sqlalchemy import select
            from models.database.knowledge_base import ChatSessionORM
            
            stmt = select(ChatSessionORM)
            result = await repos.chat_repo.session.execute(stmt)
            sessions = list(result.scalars().all())
            
            synced_count = 0
            for session in sessions:
                try:
                    await repos.chat_repo.sync_message_count(session.session_id)
                    synced_count += 1
                except Exception as e:
                    print(f"Failed to sync session {session.session_id}: {e}")
                    continue
            
            return {
                "total_sessions": len(sessions),
                "synced_sessions": synced_count,
                "message": f"Synced message counts for {synced_count}/{len(sessions)} sessions"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync all message counts: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str = Path(..., description="Chat session ID")):
    """Delete a chat session and all its messages"""
    try:
        async with get_repositories() as repos:
            # Delete messages first
            await repos.message_repo.delete_session_messages(session_id)
            
            # Delete session
            deleted = await repos.chat_repo.delete_by_id(session_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            return {"message": "Chat session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")
