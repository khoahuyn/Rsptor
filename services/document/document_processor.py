import logging
import time
from typing import Dict, Any

from models import DocumentProcessSummary
from embed.embedding import embed_texts
from config.embedding import get_embedding_settings
from database.repository_factory import get_repositories
from models.database import EmbeddingOwnerType
from utils.math_utils import token_count

# Import modular helpers (RAGFlow approach)
from .processing_helpers import DocumentProcessingHelper
from .document_utils import (
    calculate_content_hash,
    create_document_summary, 
    prepare_document_data,
    prepare_chunk_data_list,
    prepare_embedding_data_list
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    RAGFlow-style consolidated document processor - MAIN ORCHESTRATOR
    Single-pass: Parse → Chunk → Embed → Save ALL (optimal performance)
    
    Modular design using helpers and existing repositories to avoid code duplication
    """
    
    def __init__(self):
        self.embed_config = get_embedding_settings()
        # Initialize modular helpers (RAGFlow approach)
        self.processing_helper = DocumentProcessingHelper()
    
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: str,
        kb_id: str
    ) -> DocumentProcessSummary:

        start_time = time.time()
        import uuid
        doc_id = str(uuid.uuid4())
        
        logger.info(f"🚀 RAGFlow single-pass processing: {filename}")
        
        try:
            # STAGE 1: Parse + Chunk (fast in-memory) - DELEGATED TO HELPER
            chunks = await self.processing_helper.parse_and_chunk(
                file_content=file_content,
                filename=filename,
                doc_id=doc_id
            )
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            logger.info(f"📄 Generated {len(chunks)} chunks")
            
            # STAGE 2: Generate embeddings (RAGFlow direct approach)
            logger.info(f"🔢 Generating embeddings for {len(chunks)} chunks...")
            embeddings = await embed_texts(
                texts=[chunk.content for chunk in chunks],
                vector_dim=self.embed_config.embed_dimension,
                cfg=self.embed_config
            )
            logger.info(f"✅ Embeddings generated: {len(embeddings)} vectors of {len(embeddings[0]) if embeddings else 0}D")
            
            # STAGE 3: Save ALL to database (single atomic transaction) - USING HELPERS
            summary = await self._save_using_helpers_and_repos(
                doc_id=doc_id,
                filename=filename,
                file_content=file_content,
                chunks=chunks,
                embeddings=embeddings,
                tenant_id=tenant_id,
                kb_id=kb_id,
                start_time=start_time
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ RAGFlow single-pass completed in {processing_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ RAGFlow processing failed: {e}")
            raise
    
    async def _save_using_helpers_and_repos(
        self,
        doc_id: str,
        filename: str,
        file_content: bytes,
        chunks,
        embeddings,
        tenant_id: str,
        kb_id: str,
        start_time: float
    ) -> DocumentProcessSummary:
        """Save document using helpers and existing repositories (RAGFlow modular approach)"""
        
        try:
            async with get_repositories() as repos:
                # Calculate content hash using helper
                content_hash = calculate_content_hash(file_content)
                
                # STEP 1: Ensure Knowledge Base exists (use existing repository)
                await self._ensure_kb_exists_using_repo(repos, tenant_id, kb_id)
                
                # Check for existing document using existing repository method
                existing_doc = await repos.document_repo.find_by_checksum(
                    tenant_id, kb_id, content_hash
                )
                
                if existing_doc:
                    logger.info(f"📄 Document already exists: {existing_doc.doc_id}")
                    return create_document_summary(
                        doc_id=existing_doc.doc_id,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        filename=filename,
                        total_chunks=len(chunks),
                        total_embeddings=len(chunks),  # Assume 1:1 ratio for existing doc
                        start_time=start_time,
                        raptor_enabled=False  # Duplicate - no processing
                    )
                
                # ATOMIC TRANSACTION: Save document + chunks + embeddings
                logger.info("💾 Saving document + chunks + embeddings atomically...")
                
                # Prepare data using helpers
                total_tokens = sum(token_count(chunk.content) for chunk in chunks)
                document_data = prepare_document_data(
                    doc_id, tenant_id, kb_id, filename, content_hash,
                    total_tokens, len(chunks), self.processing_helper.chunking_settings
                )
                
                chunk_data_list = prepare_chunk_data_list(chunks, token_count)
                
                # Create document WITH chunks using existing repository (atomic)
                document = await repos.document_repo.create_document_with_chunks(
                    doc_data=document_data,
                    chunks_data=chunk_data_list
                )
                
                # Prepare embedding data using helper
                embedding_data_list = prepare_embedding_data_list(
                    chunks, embeddings, tenant_id, kb_id, doc_id, self.embed_config
                )
                
                # Fix EmbeddingOwnerType conversion
                for embedding_data in embedding_data_list:
                    embedding_data["owner_type"] = EmbeddingOwnerType.chunk
                
                # Bulk save embeddings using existing repository
                await repos.embedding_repo.bulk_create(embedding_data_list)
                
                # Update KB statistics using existing repository
                await self._update_kb_stats_using_repo(repos, kb_id, len(chunks), total_tokens)
                
                # Commit all changes atomically
                await repos.commit()
                
                logger.info(f"✅ Database save completed atomically (RAGFlow optimized): {len(chunks)} chunks + embeddings")
                
                return create_document_summary(
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    filename=filename,
                    total_chunks=len(chunks),
                    total_embeddings=len(embeddings),
                    start_time=start_time,
                    raptor_enabled=False  # Stage 1: basic processing only
                )
                
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    
    async def _ensure_kb_exists_using_repo(self, repos, tenant_id: str, kb_id: str) -> None:
        """Ensure Knowledge Base exists using existing repository (RAGFlow approach)"""
        try:
            # Check if KB exists using existing repo method
            existing_kb = await repos.kb_repo.get_by_id(kb_id)
            
            if existing_kb:
                logger.debug(f"📚 Using existing KB: {kb_id}")
                return
            
            # Auto-create KB using existing repo method (RAGFlow style)
            logger.info(f"🆕 Auto-creating Knowledge Base: {kb_id}")
            
            # Prepare KB settings
            kb_settings = {
                "chunk_method": "hierarchical",
                "chunk_size": self.processing_helper.chunking_settings.chunk_size,
                "chunk_overlap": self.processing_helper.chunking_settings.chunk_overlap_percent,
                "embedding_model": self.embed_config.embed_model,
                "embedding_dimension": self.embed_config.embed_dimension,
                "auto_created": True,
                "created_by": "ragflow_document_processor"
            }
            
            # Use existing repository method
            await repos.kb_repo.create_kb(
                tenant_id=tenant_id,
                name=f"Auto-created KB: {kb_id}",
                description="Auto-created by RAGFlow document processor",
                settings=kb_settings,
                kb_id=kb_id
            )
            
            logger.info(f"✅ Knowledge Base created: {kb_id}")
            
        except Exception as e:
            logger.error(f"Failed to ensure Knowledge Base exists: {e}")
            raise
    
    async def _update_kb_stats_using_repo(self, repos, kb_id: str, chunk_count: int, total_tokens: int) -> None:
        """Update KB statistics using atomic increment (race-condition safe)"""
        try:
            # Use atomic increment to avoid race conditions with RAPTOR
            await repos.kb_repo.increment_kb_stats(
                kb_id=kb_id,
                document_count_delta=1,  # Add 1 document
                chunk_count_delta=chunk_count,  # Add chunks from this document
                total_tokens_delta=total_tokens  # Add tokens from this document
            )
            logger.info(f"🔢 KB stats incremented: +1 doc, +{chunk_count} chunks, +{total_tokens} tokens")
        except Exception as e:
            logger.error(f"Failed to increment KB stats: {e}")
            raise
    
    async def get_processing_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get processing statistics for a document (delegated to helper)"""
        return await self.processing_helper.get_processing_stats(doc_id)
