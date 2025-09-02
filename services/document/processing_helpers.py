import logging
import asyncio
from typing import List, Optional, Callable, Dict, Any

from models.document import DocumentChunk
from chunking.chunk_clean import clean_content
from chunking.hierarchical_chunker import HierarchicalChunker
from config.chunking import get_chunking_settings
from utils.progress import chunking_progress_context, create_chunking_progress_callback
from utils.math_utils import token_count
from database.repository_factory import get_repositories

logger = logging.getLogger(__name__)

# Global concurrency control (RAGFlow approach)
_chunk_limiter: Optional[asyncio.Semaphore] = None


def get_chunk_limiter() -> asyncio.Semaphore:
    """Get or create global chunk processing semaphore"""
    global _chunk_limiter
    if _chunk_limiter is None:
        config = get_chunking_settings()
        _chunk_limiter = asyncio.Semaphore(config.max_concurrent_chunks)
    return _chunk_limiter


class DocumentProcessingHelper:
    """Document parsing and chunking utilities (RAGFlow modular approach)"""
    
    def __init__(self):
        self.chunking_settings = get_chunking_settings()
    
    async def parse_and_chunk(
        self,
        file_content: bytes,
        filename: str,
        doc_id: str
    ) -> List[DocumentChunk]:
        """Parse document and generate chunks (RAGFlow stage 1 + Enterprise features)"""
        
        try:
            # FEATURE 1: Concurrency Control (RAGFlow approach)
            chunk_limiter = get_chunk_limiter()
            
            async with chunk_limiter:
                logger.info(f"🔄 Processing {filename} with concurrency limit ({chunk_limiter._value}/{self.chunking_settings.max_concurrent_chunks})")
                
                # Clean and extract text content
                original_content = file_content.decode('utf-8', errors='ignore')
                clean_text = clean_content(original_content)
                
                if not clean_text or len(clean_text.strip()) < 10:
                    logger.warning("Document content too short after cleaning")
                    return []
                
                # Use hierarchical chunker for optimal chunk sizes
                chunker = HierarchicalChunker(
                    chunk_size=self.chunking_settings.chunk_size,
                    delimiter=self.chunking_settings.chunk_delimiter,
                    overlap_percent=self.chunking_settings.chunk_overlap_percent
                )
                
                # FEATURE 3: Progress Callback Integration
                progress_callback = None
                if self.chunking_settings.enable_progress_callback:
                    with chunking_progress_context(f"Chunking {filename}", show_console=False) as progress_bar:
                        progress_callback = create_chunking_progress_callback(progress_bar)
                        
                        # FEATURE 2: Thread Safety (RAGFlow approach)
                        if self.chunking_settings.enable_thread_processing and len(clean_text) > 50000:  # Large files
                            logger.info(f"⚡ Using thread processing for large file: {len(clean_text)} chars")
                            chunks = await asyncio.to_thread(
                                self._chunk_in_thread,
                                chunker, clean_text, doc_id, self.chunking_settings.chunk_size, progress_callback
                            )
                        else:
                            # Small files: direct async processing
                            chunks = await chunker.chunk_only(
                                text=clean_text,
                                doc_id=doc_id,
                                max_chunk_tokens=self.chunking_settings.chunk_size,
                                progress_callback=progress_callback
                            )
                else:
                    # No progress callback
                    if self.chunking_settings.enable_thread_processing and len(clean_text) > 50000:
                        chunks = await asyncio.to_thread(
                            self._chunk_in_thread,
                            chunker, clean_text, doc_id, self.chunking_settings.chunk_size, None
                        )
                    else:
                        chunks = await chunker.chunk_only(
                            text=clean_text,
                            doc_id=doc_id,
                            max_chunk_tokens=self.chunking_settings.chunk_size
                        )
            
            # OPTIMIZATION: Minimal chunk metadata (RAGFlow approach)
            # Common settings moved to document level to reduce storage
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    # Keep only unique per-chunk data
                    optimized_metadata = {
                        "actual_tokens": chunk.metadata.get("actual_tokens", 0),
                        "chunk_index": chunk.metadata.get("chunk_index", chunk.chunk_index)
                    }
                    chunk.metadata = optimized_metadata
                else:
                    # Fallback: minimal metadata
                    chunk.metadata = {
                        "actual_tokens": len(chunk.content.split()) * 0.75,  # Rough estimate
                        "chunk_index": chunk.chunk_index
                    }
            
            logger.info(f"📋 Chunking completed: {len(chunks)} chunks from {len(clean_text)} chars")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Parsing/chunking failed: {e}")
            raise
    
    def _chunk_in_thread(self, chunker: HierarchicalChunker, text: str, doc_id: str, 
                        max_tokens: int, progress_callback: Optional[Callable[[str], None]] = None) -> List[DocumentChunk]:
        """
        Thread-safe synchronous wrapper for chunking large files
        Called via asyncio.to_thread for heavy processing
        """
        import asyncio
        try:
            # Create new event loop for this thread (RAGFlow pattern)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run chunking in this thread's event loop
            result = loop.run_until_complete(
                chunker.chunk_only(text, doc_id, max_tokens, progress_callback)
            )
            
            return result
            
        finally:
            # Clean up loop
            try:
                loop.close()
            except:
                pass
    
    async def get_processing_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get processing statistics for a document"""
        
        try:
            async with get_repositories() as repos:
                # Get document info
                document = await repos.document_repo.get_by_id(doc_id)
                if not document:
                    return {"error": "Document not found"}
                
                # Get chunk count
                chunks = await repos.chunk_repo.get_chunks_by_doc_id(doc_id)
                
                # Get embedding count
                embeddings = await repos.embedding_repo.get_embeddings_by_doc_id(doc_id)
                
                # Include cache performance for optimization insights
                from utils import get_cache_stats
                cache_stats = get_cache_stats()
                
                return {
                    "doc_id": doc_id,
                    "filename": document.name,
                    "chunk_count": len(chunks),
                    "embedding_count": len(embeddings),
                    "total_tokens": document.token_count,
                    "status": document.status,
                    "created_at": document.created_at,
                    "cache_performance": {
                        "embed_cache_hit_rate": cache_stats.get("embed_cache", {}).get("hit_rate", "0%"),
                        "embed_cache_size": cache_stats.get("embed_cache", {}).get("size", 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {"error": str(e)}
