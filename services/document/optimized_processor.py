import logging
import time
import uuid
import asyncio
from typing import List, Tuple, Optional

from models import DocumentProcessSummary
from embed.embedding import embed_texts
from config.embedding import get_embedding_settings
from database.repository_factory import get_repositories
from models.database import EmbeddingOwnerType
from utils.math_utils import token_count

# Import existing helpers
from .processing_helpers import DocumentProcessingHelper
from .document_utils import (
    calculate_content_hash,
    create_document_summary, 
    prepare_document_data,
    prepare_chunk_data_list
)

# Import RAPTOR
from services.build_tree import build_tree

logger = logging.getLogger(__name__)


class OptimizedDocumentProcessor:

    
    def __init__(self):
        self.embed_config = get_embedding_settings()
        self.processing_helper = DocumentProcessingHelper()
    
    async def process_document_with_raptor(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: str,
        kb_id: str,
        enable_raptor: bool = True,
        max_clusters: int = 64,
        threshold: float = 0.1,
        random_seed: int = 42,
        progress_callback: Optional[callable] = None
    ) -> DocumentProcessSummary:
        """
        🚀 OPTIMIZED: Process document + RAPTOR tree in single in-memory pipeline
        """
        
        start_time = time.time()
        doc_id = str(uuid.uuid4())
        
        logger.info(f"🚀 OPTIMIZED processing START: {filename} (RAPTOR: {enable_raptor})")
        
        try:
            # 🔄 PHASE 1: Document Processing (In-Memory)
            logger.info("📄 Phase 1: Parse + Chunk + Embed (in-memory)")
            chunks, embeddings = await self._process_document_in_memory(
                file_content, filename, doc_id, progress_callback
            )
            
            original_chunks_count = len(chunks)
            
            # 🌳 PHASE 2: RAPTOR Tree Building (In-Memory)
            raptor_chunks = []
            raptor_embeddings = []
            raptor_levels = []  # ✅ Initialize for when raptor is disabled
            tree_levels = 0
            
            if enable_raptor and len(chunks) > 1:
                logger.info("🌳 Phase 2: RAPTOR tree building (in-memory)")
                if progress_callback:
                    progress_callback("Building RAPTOR tree in memory...")
                
                raptor_chunks, raptor_embeddings, raptor_levels, tree_levels = await self._build_raptor_in_memory(
                    chunks, embeddings, max_clusters, threshold, random_seed, progress_callback
                )
            
            # 💾 PHASE 3: Save Everything (Single Transaction)
            logger.info("💾 Phase 3: Save ALL data (single atomic transaction)")
            if progress_callback:
                progress_callback("Saving all data to database...")
            
            summary = await self._save_all_data_atomically(
                doc_id=doc_id,
                filename=filename,
                file_content=file_content,
                original_chunks=chunks,
                original_embeddings=embeddings,
                raptor_chunks=raptor_chunks,
                raptor_embeddings=raptor_embeddings,
                raptor_levels=raptor_levels if enable_raptor else [],  # ✅ Fix Issue 3: Pass level info
                tenant_id=tenant_id,
                kb_id=kb_id,
                tree_levels=tree_levels,
                enable_raptor=enable_raptor,
                start_time=start_time
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ OPTIMIZED processing completed in {processing_time:.2f}s")
            logger.info(f"📊 Results: {original_chunks_count} original + {len(raptor_chunks)} RAPTOR = {original_chunks_count + len(raptor_chunks)} total chunks")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ OPTIMIZED processing failed: {e}")
            raise
    
    async def _process_document_in_memory(
        self, 
        file_content: bytes, 
        filename: str, 
        doc_id: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List, List[List[float]]]:
        """Process document and return chunks + embeddings in memory"""
        
        # STAGE 1: Parse + Chunk
        if progress_callback:
            progress_callback("Parsing and chunking document...")
        
        chunks = await self.processing_helper.parse_and_chunk(
            file_content=file_content,
            filename=filename,
            doc_id=doc_id
        )
        
        if not chunks:
            raise ValueError("No chunks generated from document")
        
        logger.info(f"📄 Generated {len(chunks)} chunks")
        
        # STAGE 2: Generate embeddings
        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...")
        
        logger.info(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = await embed_texts(
            texts=[chunk.content for chunk in chunks],
            vector_dim=self.embed_config.embed_dimension,
            cfg=self.embed_config
        )
        logger.info(f"✅ Embeddings generated: {len(embeddings)} vectors of {len(embeddings[0]) if embeddings else 0}D")
        
        return chunks, embeddings
    
    async def _build_raptor_in_memory(
        self,
        chunks: List,
        embeddings: List[List[float]],
        max_clusters: int,
        threshold: float, 
        random_seed: int,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[str], List[List[float]], List[int], int]:
        """Build RAPTOR tree in memory and return summary chunks + embeddings"""
        
        import numpy as np
        
        # Prepare data for RAPTOR (convert to expected format)
        chunks_with_embeddings = [
            (chunk.content, np.array(emb)) for chunk, emb in zip(chunks, embeddings)
        ]
        
        logger.info(f"🌳 Starting RAPTOR tree building: {len(chunks_with_embeddings)} chunks")
        
        # Build RAPTOR tree using existing optimized code
        tree_result = await build_tree(
            chunks=chunks_with_embeddings,
            max_clusters=max_clusters,
            threshold=threshold,
            random_seed=random_seed,
            callback=progress_callback
        )
        
        # Extract RAPTOR summaries (exclude original chunks)
        augmented_chunks = tree_result.get('augmented_chunks', [])
        original_count = tree_result.get('original_count', len(chunks))
        
        raptor_summaries = []
        raptor_embeddings = []
        raptor_levels = []  # ✅ Fix Issue 3: Preserve level info
        
        # Get only the NEW summary chunks (not original chunks)
        for i, (content, embedding, level) in enumerate(augmented_chunks):
            if i >= original_count:  # Only summary chunks
                raptor_summaries.append(content)
                raptor_embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
                raptor_levels.append(level)  # ✅ Store actual level
        
        tree_levels = tree_result.get('tree_levels', 0)
        
        logger.info(f"✅ RAPTOR tree completed: {len(raptor_summaries)} summaries, {tree_levels} levels")
        
        return raptor_summaries, raptor_embeddings, raptor_levels, tree_levels
    
    async def _save_all_data_atomically(
        self,
        doc_id: str,
        filename: str,
        file_content: bytes,
        original_chunks: List,
        original_embeddings: List[List[float]],
        raptor_chunks: List[str],
        raptor_embeddings: List[List[float]],
        raptor_levels: List[int],  # ✅ Fix Issue 3: Add level info
        tenant_id: str,
        kb_id: str,
        tree_levels: int,
        enable_raptor: bool,
        start_time: float
    ) -> DocumentProcessSummary:
        """Save all data in single atomic transaction"""
        
        try:
            async with get_repositories() as repos:
                # Check for duplicates
                content_hash = calculate_content_hash(file_content)
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
                        total_chunks=len(original_chunks),
                        total_embeddings=len(original_chunks),
                        start_time=start_time,
                        raptor_enabled=False
                    )
                
                # Ensure KB exists
                await self._ensure_kb_exists_using_repo(repos, tenant_id, kb_id)
                
                # Prepare document data
                total_tokens = sum(token_count(chunk.content) for chunk in original_chunks)
                if raptor_chunks:
                    total_tokens += sum(token_count(chunk) for chunk in raptor_chunks)
                
                document_data = prepare_document_data(
                    doc_id, tenant_id, kb_id, filename, content_hash,
                    total_tokens, len(original_chunks) + len(raptor_chunks), 
                    self.processing_helper.chunking_settings
                )
                
                # Prepare ALL chunk data (original + RAPTOR)
                chunk_data_list = prepare_chunk_data_list(original_chunks, token_count)
                
                # Add RAPTOR chunks with generated chunk_id
                for i, raptor_chunk in enumerate(raptor_chunks):
                    raptor_chunk_id = f"{doc_id}_chunk_{len(original_chunks) + i}"
                    actual_level = raptor_levels[i] if i < len(raptor_levels) else 1  # ✅ Fix Issue 3: Use actual level
                    chunk_data_list.append({
                        "chunk_id": raptor_chunk_id,  # Add required chunk_id
                        "doc_id": doc_id,
                        "chunk_index": len(original_chunks) + i,
                        "content": raptor_chunk,
                        "token_count": token_count(raptor_chunk),
                        "meta": {
                            "type": "raptor_summary",
                            "level": actual_level,  # ✅ Fix Issue 3: Use actual level from build_tree
                            "created_by": "raptor_builder"
                        }
                    })
                
                # 🚀 ATOMIC SAVE: Document + ALL chunks
                document = await repos.document_repo.create_document_with_chunks(
                    doc_data=document_data,
                    chunks_data=chunk_data_list
                )
                
                # Prepare ALL embedding data (original + RAPTOR)
                all_embeddings = original_embeddings + raptor_embeddings
                all_chunk_count = len(original_chunks) + len(raptor_chunks)
                
                embedding_data_list = []
                
                # Create embedding data for original chunks
                for i, embedding in enumerate(original_embeddings):
                    chunk_id = original_chunks[i].chunk_id
                    embedding_id = f"{chunk_id}_embedding"
                    embedding_data_list.append({
                        "id": embedding_id,  # Add required embedding ID
                        "owner_id": chunk_id,
                        "owner_type": EmbeddingOwnerType.chunk,
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "vector": embedding,
                        "model": self.embed_config.embed_model,
                        "dimension": len(embedding),
                        "meta": {
                            "doc_id": doc_id,
                            "type": "original_chunk"
                        }
                    })
                
                # Create embedding data for RAPTOR chunks
                for i, embedding in enumerate(raptor_embeddings):
                    raptor_chunk_id = f"{doc_id}_chunk_{len(original_chunks) + i}"
                    embedding_id = f"{raptor_chunk_id}_embedding"
                    actual_level = raptor_levels[i] if i < len(raptor_levels) else 1
                    
                    # ✅ Fix Issue 2: Use correct owner_type based on level
                    owner_type = EmbeddingOwnerType.root if actual_level >= tree_levels else EmbeddingOwnerType.summary
                    
                    embedding_data_list.append({
                        "id": embedding_id,  # Add required embedding ID
                        "owner_id": raptor_chunk_id,
                        "owner_type": owner_type,  # ✅ Fix Issue 2: Use summary/root instead of chunk
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "vector": embedding,
                        "model": self.embed_config.embed_model,
                        "dimension": len(embedding),
                        "meta": {
                            "doc_id": doc_id,
                            "type": "raptor_summary",
                            "level": actual_level  # ✅ Fix Issue 3: Add level to embedding meta
                        }
                    })
                
                # 🚀 BULK SAVE: ALL embeddings in smaller chunks to avoid timeout
                logger.info(f"💾 Saving {len(embedding_data_list)} embeddings in chunks...")
                
                # Split into smaller chunks (reduced from 50 to 20 for faster refresh)
                chunk_size = 20  # ✅ Reduce chunk size to avoid refresh timeouts
                for i in range(0, len(embedding_data_list), chunk_size):
                    chunk = embedding_data_list[i:i + chunk_size]
                    logger.info(f"💾 Saving embedding chunk {i//chunk_size + 1}/{(len(embedding_data_list) + chunk_size - 1)//chunk_size}: {len(chunk)} embeddings")
                    
                    try:
                        await asyncio.wait_for(
                            repos.embedding_repo.bulk_create(chunk, skip_refresh=True),  # ✅ Skip refresh: embeddings have pre-defined IDs
                            timeout=30.0  # ✅ Reduced back to 30s: skip_refresh eliminates slow individual queries
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Timeout saving embedding chunk {i//chunk_size + 1}")
                        raise ValueError(f"Database save timeout for embedding chunk {i//chunk_size + 1}")
                    except Exception as e:
                        logger.error(f"❌ Error saving embedding chunk {i//chunk_size + 1}: {e}")
                        raise
                
                # Update KB stats with timeout
                logger.info(f"📊 Updating KB stats: 1 document, {all_chunk_count} chunks, {total_tokens} tokens")
                try:
                    await asyncio.wait_for(
                        repos.kb_repo.increment_kb_stats(
                            kb_id=kb_id,
                            document_count_delta=1,  # ✅ Fix Issue 1: Add document count
                            chunk_count_delta=all_chunk_count,
                            total_tokens_delta=total_tokens
                        ),
                        timeout=20.0  # ✅ Increased timeout: 10s -> 20s for consistency
                    )
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout updating KB stats")
                    raise ValueError(f"Database timeout updating KB stats")
                except Exception as e:
                    logger.error(f"❌ Error updating KB stats: {e}")
                    raise
                
                logger.info(f"✅ ATOMIC SAVE completed: {all_chunk_count} chunks + {len(all_embeddings)} embeddings")
                
                # Create summary
                processing_time = time.time() - start_time
                return DocumentProcessSummary(
                    doc_id=doc_id,
                    filename=filename,
                    total_chunks=all_chunk_count,
                    total_embeddings=len(all_embeddings),
                    processing_time=processing_time,
                    status="completed",
                    original_chunks=len(original_chunks),
                    summary_chunks=len(raptor_chunks),
                    raptor_enabled=enable_raptor,
                    raptor_summary_count=len(raptor_chunks),
                    raptor_tree_levels=tree_levels,
                    tenant_id=tenant_id,
                    kb_id=kb_id
                )
                
        except Exception as e:
            logger.error(f"❌ Atomic save failed: {e}")
            raise
    
    async def _ensure_kb_exists_using_repo(self, repos, tenant_id: str, kb_id: str):
        """Ensure knowledge base exists using repository"""
        try:
            existing_kb = await repos.kb_repo.get_by_id(kb_id)
            if existing_kb:
                logger.info(f"✅ Knowledge base exists: {tenant_id}::{kb_id}")
            else:
                raise ValueError("KB not found")
        except Exception:
            logger.info(f"📝 Creating knowledge base: {tenant_id}::{kb_id}")
            try:
                await repos.kb_repo.create_kb(
                    tenant_id=tenant_id,
                    name=kb_id,
                    description=f"Auto-created KB for {kb_id}",
                    kb_id=kb_id  # Pass kb_id as optional parameter
                )
                logger.info(f"✅ Knowledge base created: {tenant_id}::{kb_id}")
            except Exception as create_error:
                logger.error(f"❌ Failed to create KB {tenant_id}::{kb_id}: {create_error}")
                raise


# Global instance for easy access
_optimized_processor = None

def get_optimized_processor() -> OptimizedDocumentProcessor:
    """Get global optimized processor instance"""
    global _optimized_processor
    if _optimized_processor is None:
        _optimized_processor = OptimizedDocumentProcessor()
    return _optimized_processor
