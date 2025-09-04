from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from .base import BaseRepository
from models.database.message import MessageORM, MessageRole


class MessageRepository(BaseRepository[MessageORM]):
    """Repository for chat message management"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MessageORM)
    
    async def create_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        message_id: Optional[str] = None
    ) -> MessageORM:
        """Create a new message in a chat session"""
        import uuid
        
        # Generate message_id if not provided
        if message_id is None:
            message_id = f"msg::{uuid.uuid4().hex[:12]}"
        
        message_data = {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "extra_metadata": extra_metadata or {},
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        return await self.create(**message_data)
    
    async def get_session_messages(
        self, 
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        include_system: bool = True
    ) -> List[MessageORM]:
        """Get messages for a chat session with pagination"""
        try:
            stmt = select(MessageORM).where(MessageORM.session_id == session_id)
            
            # Optionally exclude system messages
            if not include_system:
                stmt = stmt.where(MessageORM.role != MessageRole.system)
            
            # Order by creation time (oldest first)
            stmt = stmt.order_by(MessageORM.created_at)
            
            # Apply pagination
            if limit > 0:
                stmt = stmt.limit(limit).offset(offset)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise ValueError(f"Failed to get session messages: {str(e)}")
    
    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 10
    ) -> List[MessageORM]:
        """Get most recent messages for context building"""
        try:
            stmt = (
                select(MessageORM)
                .where(MessageORM.session_id == session_id)
                .order_by(desc(MessageORM.created_at))
                .limit(count)
            )
            
            result = await self.session.execute(stmt)
            messages = list(result.scalars().all())
            
            # Return in chronological order (oldest first)
            return list(reversed(messages))
        except Exception as e:
            raise ValueError(f"Failed to get recent messages: {str(e)}")
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> str:
        """Build conversation context string for LLM"""
        try:
            messages = await self.get_recent_messages(session_id, max_messages)
            
            context_parts = []
            for msg in messages:
                if msg.role == MessageRole.system:
                    continue  # Skip system messages in context
                
                role_label = "User" if msg.role == MessageRole.user else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            raise ValueError(f"Failed to build conversation context: {str(e)}")
    
    async def update_message_tokens(
        self,
        message_id: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> Optional[MessageORM]:
        """Update token usage for a message"""
        update_data = {}
        if input_tokens is not None:
            update_data["input_tokens"] = input_tokens
        if output_tokens is not None:
            update_data["output_tokens"] = output_tokens
        
        if update_data:
            return await self.update_by_id(message_id, **update_data)
        return None
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a chat session"""
        try:
            messages = await self.get_session_messages(session_id, limit=0)  # Get all
            
            total_messages = len(messages)
            user_messages = len([m for m in messages if m.role == MessageRole.user])
            assistant_messages = len([m for m in messages if m.role == MessageRole.assistant])
            
            total_input_tokens = sum(m.input_tokens or 0 for m in messages)
            total_output_tokens = sum(m.output_tokens or 0 for m in messages)
            
            return {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        except Exception as e:
            raise ValueError(f"Failed to get session stats: {str(e)}")
    
    async def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session"""
        try:
            messages = await self.get_session_messages(session_id, limit=0)
            count = len(messages)
            
            for message in messages:
                await self.delete_by_id(message.message_id)
            
            return count
        except Exception as e:
            raise ValueError(f"Failed to delete session messages: {str(e)}")
    
    async def search_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[MessageORM]:
        """Search messages by content within a session"""
        try:
            stmt = (
                select(MessageORM)
                .where(
                    MessageORM.session_id == session_id,
                    MessageORM.content.ilike(f"%{query}%")
                )
                .order_by(desc(MessageORM.created_at))
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise ValueError(f"Failed to search messages: {str(e)}")
