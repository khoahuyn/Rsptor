from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import insert

from .base import BaseRepository
from models.database.document import DocumentORM, ChunkORM


class DocumentRepository(BaseRepository[DocumentORM]):
    """Repository for document and chunk operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, DocumentORM)
    
    async def create_document_with_chunks(
        self, 
        doc_data: Dict[str, Any], 
        chunks_data: List[Dict[str, Any]]
    ) -> DocumentORM:
        """Create document with associated chunks in single transaction"""
        try:
            # Create document
            document = await self.create(**doc_data)
            
            # Add doc_id to all chunks
            for chunk_data in chunks_data:
                chunk_data["doc_id"] = document.doc_id
            
            # Create chunks
            if chunks_data:
                chunk_objects = [ChunkORM(**chunk_data) for chunk_data in chunks_data]
                self.session.add_all(chunk_objects)
                await self.session.flush()
            
            # Reload document with chunks
            return await self.get_document_with_chunks(document.doc_id)
            
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to create document with chunks: {str(e)}")
    
    async def get_document_with_chunks(self, doc_id: str) -> Optional[DocumentORM]:
        """Get document with all associated chunks"""
        try:
            stmt = (
                select(DocumentORM)
                .options(selectinload(DocumentORM.chunks))
                .where(DocumentORM.doc_id == doc_id)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to get document with chunks: {str(e)}")
    
    async def find_by_checksum(
        self, 
        tenant_id: str, 
        kb_id: str, 
        checksum: str
    ) -> Optional[DocumentORM]:
        """Find document by checksum within tenant/kb scope"""
        try:
            stmt = select(DocumentORM).where(
                DocumentORM.tenant_id == tenant_id,
                DocumentORM.kb_id == kb_id,
                DocumentORM.checksum == checksum
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to find document by checksum: {str(e)}")
    
    async def find_by_filename(
        self, 
        tenant_id: str, 
        kb_id: str, 
        filename: str
    ) -> Optional[DocumentORM]:
        """Find document by filename within tenant/kb scope"""
        try:
            stmt = select(DocumentORM).where(
                DocumentORM.tenant_id == tenant_id,
                DocumentORM.kb_id == kb_id,
                DocumentORM.filename == filename
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to find document by filename: {str(e)}")
    
    async def get_documents_by_tenant_kb(
        self, 
        tenant_id: str, 
        kb_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentORM]:
        """Get documents for specific tenant and knowledge base"""
        return await self.get_many(
            limit=limit,
            offset=offset,
            tenant_id=tenant_id,
            kb_id=kb_id
        )
    
    async def update_document_content(
        self, 
        doc_id: str, 
        content: str, 
        processing_stats: Optional[Dict[str, Any]] = None
    ) -> Optional[DocumentORM]:
        """Update document content and processing stats"""
        update_data = {"content": content}
        if processing_stats:
            update_data["processing_stats"] = processing_stats
        
        return await self.update_by_id(doc_id, **update_data)


class ChunkRepository(BaseRepository[ChunkORM]):
    """Repository for chunk-specific operations"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ChunkORM)
    
    async def get_chunks_by_document(
        self, 
        doc_id: str,
        order_by_index: bool = True
    ) -> List[ChunkORM]:
        """Get all chunks for a document, optionally ordered by index"""
        try:
            stmt = select(ChunkORM).where(ChunkORM.doc_id == doc_id)
            
            if order_by_index:
                stmt = stmt.order_by(ChunkORM.chunk_index)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise ValueError(f"Failed to get chunks by document: {str(e)}")
    
    async def bulk_upsert_chunks(self, chunks_data: List[Dict[str, Any]]) -> int:
        """Bulk upsert chunks with conflict resolution"""
        try:
            if not chunks_data:
                return 0
            
            stmt = insert(ChunkORM).values(chunks_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ChunkORM.chunk_id],
                set_={
                    "content": stmt.excluded.content,
                    "token_count": stmt.excluded.token_count,
                    "meta": stmt.excluded.meta,
                    "chunk_index": stmt.excluded.chunk_index,
                    "doc_id": stmt.excluded.doc_id,
                }
            )
            result = await self.session.execute(stmt)
            return len(chunks_data)
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to bulk upsert chunks: {str(e)}")
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkORM]:
        """Get chunk with document information"""
        try:
            stmt = (
                select(ChunkORM)
                .options(selectinload(ChunkORM.document))
                .where(ChunkORM.chunk_id == chunk_id)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise ValueError(f"Failed to get chunk by ID: {str(e)}")
    
    async def search_chunks_by_content(
        self,
        tenant_id: str,
        kb_id: str, 
        query: str,
        limit: int = 10
    ) -> List[ChunkORM]:
        """Simple text search in chunk content (can be enhanced with full-text search later)"""
        try:
            stmt = (
                select(ChunkORM)
                .join(ChunkORM.document)
                .where(
                    DocumentORM.tenant_id == tenant_id,
                    DocumentORM.kb_id == kb_id,
                    ChunkORM.content.ilike(f"%{query}%")
                )
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise ValueError(f"Failed to search chunks: {str(e)}")



