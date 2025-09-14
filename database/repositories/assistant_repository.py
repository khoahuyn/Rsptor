from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .base import BaseRepository
from models.database.assistant import AssistantORM


class AssistantRepository(BaseRepository[AssistantORM]):
    """Repository for AI Assistant management"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AssistantORM)
    
    async def create_assistant(
        self,
        tenant_id: str,
        kb_id: str,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        avatar_url: Optional[str] = None,
        assistant_id: Optional[str] = None
    ) -> AssistantORM:
        """Create a new AI assistant"""
        import uuid
        
        # Generate assistant_id if not provided
        if assistant_id is None:
            assistant_id = f"{tenant_id}::assistant::{uuid.uuid4().hex[:8]}"
        
        # Default model settings
        if model_settings is None:
            model_settings = {
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_k": 8,
                "use_rerank": True
            }
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."
        
        assistant_data = {
            "assistant_id": assistant_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "model_settings": model_settings,
            "avatar_url": avatar_url
        }
        
        return await self.create(**assistant_data)
    
    async def get_assistants_by_tenant(
        self, 
        tenant_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[AssistantORM]:
        """Get all assistants for a tenant with pagination"""
        return await self.get_many(
            limit=limit,
            offset=offset,
            tenant_id=tenant_id
        )
    
    async def get_assistant_with_kb(self, assistant_id: str) -> Optional[AssistantORM]:
        """Get assistant with knowledge base details"""
        try:
            stmt = (
                select(AssistantORM)
                .options(selectinload(AssistantORM.knowledge_base))
                .where(AssistantORM.assistant_id == assistant_id)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to get assistant with KB: {str(e)}")
    
    async def get_assistants_by_kb(self, kb_id: str) -> List[AssistantORM]:
        """Get all assistants for a specific knowledge base"""
        return await self.get_many(kb_id=kb_id)
    
    async def update_assistant(
        self,
        assistant_id: str,
        **updates
    ) -> Optional[AssistantORM]:
        """Update assistant properties"""
        return await self.update_by_id(assistant_id, **updates)
    
    async def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant (and cascade to chat sessions)"""
        return await self.delete_by_id(assistant_id)
    
    async def get_assistant_stats(self, assistant_id: str) -> Optional[Dict[str, Any]]:
        """Get assistant statistics (chat sessions, messages, etc.)"""
        try:
            assistant = await self.get_assistant_with_kb(assistant_id)
            if not assistant:
                return None
                
            # Count chat sessions for this assistant
            from models.database.knowledge_base import ChatSessionORM
            stmt = select(ChatSessionORM).where(ChatSessionORM.assistant_id == assistant_id)
            result = await self.session.execute(stmt)
            sessions = result.scalars().all()
            
            total_messages = sum(session.message_count for session in sessions)
            
            return {
                "assistant_id": assistant.assistant_id,
                "name": assistant.name,
                "kb_name": assistant.knowledge_base.name if assistant.knowledge_base else None,
                "total_sessions": len(sessions),
                "total_messages": total_messages,
                "created_at": assistant.created_at,
                "last_updated": assistant.updated_at
            }
        except Exception as e:
            raise ValueError(f"Failed to get assistant stats: {str(e)}")
    
    async def search_assistants(
        self,
        tenant_id: str, 
        query: str,
        limit: int = 10
    ) -> List[AssistantORM]:
        """Search assistants by name or description"""
        try:
            stmt = (
                select(AssistantORM)
                .where(
                    AssistantORM.tenant_id == tenant_id,
                    (AssistantORM.name.ilike(f"%{query}%")) | 
                    (AssistantORM.description.ilike(f"%{query}%"))
                )
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise ValueError(f"Failed to search assistants: {str(e)}")
