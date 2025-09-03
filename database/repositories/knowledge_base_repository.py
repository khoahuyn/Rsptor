from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, update, select
from .base import BaseRepository
from models.database.knowledge_base import KnowledgeBaseORM, ChatSessionORM, KnowledgeBaseStatus
from utils.kb_settings import get_default_kb_settings


class KnowledgeBaseRepository(BaseRepository[KnowledgeBaseORM]):
    """Repository for Knowledge Base management"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, KnowledgeBaseORM)
    
    async def create_kb(
        self,
        tenant_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        kb_id: Optional[str] = None
    ) -> KnowledgeBaseORM:
        """Create a new knowledge base"""
        # Use provided kb_id or generate from tenant_id + name
        if kb_id is None:
            kb_id = f"{tenant_id}::kb::{name.lower().replace(' ', '_')}"
        
        kb_data = {
            "kb_id": kb_id,
            "tenant_id": tenant_id,
            "name": name,
            "description": description,
            "settings": settings or get_default_kb_settings(),
            "status": KnowledgeBaseStatus.active
        }
        
        return await self.create(**kb_data)
    
    async def get_kbs_by_tenant(
        self, 
        tenant_id: str,
        status: Optional[KnowledgeBaseStatus] = None
    ) -> List[KnowledgeBaseORM]:
        """Get all knowledge bases for a tenant"""
        filters = {"tenant_id": tenant_id}
        if status:
            filters["status"] = status
            
        return await self.get_many(**filters)
    
    async def update_kb_stats(
        self,
        kb_id: str,
        document_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        total_tokens: Optional[int] = None
    ) -> Optional[KnowledgeBaseORM]:
        """Update KB statistics after document processing"""
        update_data = {}
        if document_count is not None:
            update_data["document_count"] = document_count
        if chunk_count is not None:
            update_data["chunk_count"] = chunk_count
        if total_tokens is not None:
            update_data["total_tokens"] = total_tokens
            
        return await self.update_by_id(kb_id, **update_data)
    
    async def increment_kb_stats(
        self,
        kb_id: str,
        document_count_delta: int = 0,
        chunk_count_delta: int = 0, 
        total_tokens_delta: int = 0
    ) -> Optional[KnowledgeBaseORM]:
        """Atomically increment KB statistics (race-condition safe)"""
        from sqlalchemy import text
        
        # Use raw SQL for atomic increment operations
        query = text("""
            UPDATE knowledge_bases 
            SET 
                document_count = document_count + :doc_delta,
                chunk_count = chunk_count + :chunk_delta,
                total_tokens = total_tokens + :tokens_delta,
                updated_at = NOW()
            WHERE kb_id = :kb_id
            RETURNING *
        """)
        
        result = await self.session.execute(query, {
            "kb_id": kb_id,
            "doc_delta": document_count_delta,
            "chunk_delta": chunk_count_delta, 
            "tokens_delta": total_tokens_delta
        })
        
        row = result.fetchone()
        return KnowledgeBaseORM(**row._asdict()) if row else None
    
    async def get_kb_statistics(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a knowledge base"""
        try:
            # This would join with documents, chunks, trees tables for real stats
            # For now, return basic stats from KB record
            kb = await self.get_by_id(kb_id)
            if not kb:
                return None
                
            return {
                "kb_id": kb.kb_id,
                "name": kb.name,
                "status": kb.status.value,
                "document_count": kb.document_count,
                "chunk_count": kb.chunk_count,
                "total_tokens": kb.total_tokens,
                "created_at": kb.created_at,
                "updated_at": kb.updated_at
            }
        except Exception as e:
            raise ValueError(f"Failed to get KB statistics: {str(e)}")


class ChatSessionRepository(BaseRepository[ChatSessionORM]):
    """Repository for chat session management"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ChatSessionORM)
    
    async def create_chat_session(
        self,
        tenant_id: str,
        kb_id: str,
        name: str,
        system_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> ChatSessionORM:
        """Create a new chat session linked to a KB"""
        import uuid
        session_id = f"{tenant_id}::chat::{uuid.uuid4().hex[:8]}"
        
        session_data = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "name": name,
            "system_prompt": system_prompt,
            "settings": settings or {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_k": 5
            }
        }
        
        return await self.create(**session_data)
    
    async def get_sessions_by_tenant(
        self, 
        tenant_id: str,
        kb_id: Optional[str] = None
    ) -> List[ChatSessionORM]:
        """Get chat sessions for a tenant, optionally filtered by KB"""
        filters = {"tenant_id": tenant_id}
        if kb_id:
            filters["kb_id"] = kb_id
            
        return await self.get_many(**filters)
    
    async def get_sessions_by_kb(self, kb_id: str) -> List[ChatSessionORM]:
        """Get all chat sessions for a specific knowledge base"""
        return await self.get_many(kb_id=kb_id)
    
    async def increment_message_count(self, session_id: str, count: int = 1) -> Optional[ChatSessionORM]:
        """Increment message count and update last active"""
        try:
            stmt = update(ChatSessionORM).where(
                ChatSessionORM.session_id == session_id
            ).values(
                message_count=ChatSessionORM.message_count + count,
                last_active=func.now()
            )
            await self.session.execute(stmt)
            return await self.get_by_id(session_id)
        except Exception as e:
            raise ValueError(f"Failed to increment message count: {str(e)}")
    
    async def sync_message_count(self, session_id: str) -> Optional[ChatSessionORM]:
        """Sync message count with actual messages in database"""
        try:
            # Count actual messages in database
            from models.database.message import MessageORM
            stmt = select(func.count(MessageORM.message_id)).where(MessageORM.session_id == session_id)
            result = await self.session.execute(stmt)
            actual_count = result.scalar() or 0
            
            # Update session with actual count
            stmt = update(ChatSessionORM).where(
                ChatSessionORM.session_id == session_id
            ).values(
                message_count=actual_count,
                last_active=func.now()
            )
            await self.session.execute(stmt)
            return await self.get_by_id(session_id)
        except Exception as e:
            raise ValueError(f"Failed to sync message count: {str(e)}")
