from typing import Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure environment variables are loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .connection import get_cached_session_factory
from .repositories import (
    DocumentRepository, ChunkRepository, 
    EmbeddingRepository, 
    KnowledgeBaseRepository, ChatSessionRepository
)


class RepositoryFactory:
    """Factory for creating repository instances with shared session"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._document_repo: Optional[DocumentRepository] = None
        self._chunk_repo: Optional[ChunkRepository] = None

        self._embedding_repo: Optional[EmbeddingRepository] = None
        self._kb_repo: Optional[KnowledgeBaseRepository] = None
        self._chat_repo: Optional[ChatSessionRepository] = None
    
    @property
    def document_repo(self) -> DocumentRepository:
        if self._document_repo is None:
            self._document_repo = DocumentRepository(self.session)
        return self._document_repo
    
    @property
    def chunk_repo(self) -> ChunkRepository:
        if self._chunk_repo is None:
            self._chunk_repo = ChunkRepository(self.session)
        return self._chunk_repo
    
    
    @property
    def embedding_repo(self) -> EmbeddingRepository:
        if self._embedding_repo is None:
            self._embedding_repo = EmbeddingRepository(self.session)
        return self._embedding_repo
    
    @property
    def kb_repo(self) -> KnowledgeBaseRepository:
        if self._kb_repo is None:
            self._kb_repo = KnowledgeBaseRepository(self.session)
        return self._kb_repo
    
    @property
    def chat_repo(self) -> ChatSessionRepository:
        if self._chat_repo is None:
            self._chat_repo = ChatSessionRepository(self.session)
        return self._chat_repo
    
    async def commit(self):
        """Commit all changes in current session"""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback all changes in current session"""
        await self.session.rollback()
    
    async def close(self):
        """Close the session"""
        await self.session.close()


@asynccontextmanager
async def get_repositories():
    """Context manager for repository factory with auto-cleanup and caching"""
    session_factory = get_cached_session_factory()
    async with session_factory() as session:
        factory = RepositoryFactory(session)
        try:
            yield factory
        except Exception:
            await factory.rollback()
            raise
        else:
            await factory.commit()


# Convenience functions for common operations
async def with_repositories(func, *args, **kwargs):
    """Execute function with repository factory"""
    async with get_repositories() as repos:
        return await func(repos, *args, **kwargs)
