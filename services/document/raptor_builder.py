import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Callable

from services.build_tree import build_tree
from database.repository_factory import get_repositories
from models.database import EmbeddingOwnerType
from config.embedding import get_embedding_settings


logger = logging.getLogger(__name__)


class RaptorBuilder:
    """
    Separate RAPTOR tree building stage (RAGFlow approach)
    Runs AFTER document + chunks + embeddings are saved
    """
    
    def __init__(self):
        self.embed_config = get_embedding_settings()
    
    async def build_raptor_tree(
        self,
        doc_id: str,
        tenant_id: str,
        kb_id: str,
        max_clusters: int = None, 
        threshold: float = None,    
        random_seed: int = None,   
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:

        start_time = time.time()
        
        logger.info(f"🌳 Building RAPTOR tree for document {doc_id}")
        
        # Load config defaults if not provided (consistent with BuildTree approach)
        from config.raptor import get_raptor_settings
        raptor_config = get_raptor_settings()
        
        max_clusters = max_clusters or raptor_config.max_clusters
        threshold = threshold or raptor_config.similarity_threshold  
        random_seed = random_seed or raptor_config.random_seed
        
        logger.info(f"🎯 RAPTOR params: clusters={max_clusters}, threshold={threshold}, seed={random_seed}")
        
        try:
            # STAGE 1: Load existing chunks with embeddings
            if progress_callback:
                progress_callback("Loading chunks and embeddings from database")
                
            chunks_with_embeddings = await self._load_chunks_with_embeddings(
                doc_id, tenant_id, kb_id
            )
            
            if not chunks_with_embeddings:
                raise ValueError("No chunks found for RAPTOR tree building")
            
            logger.info(f"📦 Loaded {len(chunks_with_embeddings)} chunks with embeddings")
            
            if progress_callback:
                progress_callback(f"Starting RAPTOR clustering for {len(chunks_with_embeddings)} chunks")
            
            # STAGE 2: Build RAPTOR tree (clustering + summarization)
            tree_result = await build_tree(
                chunks=chunks_with_embeddings,
                max_clusters=max_clusters,
                threshold=threshold,
                random_seed=random_seed,
                callback=progress_callback
            )
            
            if not tree_result.get('augmented_chunks'):
                logger.warning("No RAPTOR summaries generated")
                return {
                    "status": "no_summaries",
                    "message": "Document too small for RAPTOR clustering",
                    "original_chunks": len(chunks_with_embeddings),
                    "summary_chunks": 0,
                    "total_chunks": len(chunks_with_embeddings),  # ✅ Same as original (no summaries)
                    "processing_time": time.time() - start_time
                }
            
            augmented_chunks = tree_result['augmented_chunks']
            logger.info(f"🎯 Generated {len(augmented_chunks)} RAPTOR summary chunks")
            
            # STAGE 3: Save ONLY new summary chunks + embeddings
            if progress_callback:
                progress_callback(f"Saving {len(augmented_chunks) - len(chunks_with_embeddings)} summary chunks to database")
                
            await self._save_summary_chunks(
                augmented_chunks=augmented_chunks,
                original_count=len(chunks_with_embeddings),  # ⚡ Pass count to avoid duplicate query
                doc_id=doc_id,
                tenant_id=tenant_id,
                kb_id=kb_id
            )
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(f"RAPTOR tree completed in {processing_time:.2f}s")
            
            # Get cache performance stats for optimization insights
            from utils import get_cache_stats
            cache_stats = get_cache_stats()
            
            logger.info(f"✅ RAPTOR tree completed in {processing_time:.2f}s")
            logger.info(f"📊 Cache performance - LLM: {cache_stats.get('llm_cache', {}).get('hit_rate', '0%')}, Embed: {cache_stats.get('embed_cache', {}).get('hit_rate', '0%')}")
            
            return {
                "status": "completed",
                "doc_id": doc_id,
                "original_chunks": tree_result.get('original_count', len(chunks_with_embeddings)),
                "summary_chunks": tree_result.get('summary_count', 0),  # ✅ Only RAPTOR summaries
                "total_chunks": tree_result.get('total_count', len(augmented_chunks)),  # ✅ From build_tree
                "tree_levels": tree_result.get('tree_levels', 0),
                "processing_time": processing_time,
                "cache_performance": {
                    "llm_cache_hit_rate": cache_stats.get("llm_cache", {}).get("hit_rate", "0%"),
                    "embed_cache_hit_rate": cache_stats.get("embed_cache", {}).get("hit_rate", "0%"),
                    "llm_cache_size": cache_stats.get("llm_cache", {}).get("size", 0),
                    "embed_cache_size": cache_stats.get("embed_cache", {}).get("size", 0)
                },
                "config": {
                    "max_clusters": max_clusters,
                    "threshold": threshold, 
                    "random_seed": random_seed,
                    "source": "config defaults"  # Indicate values came from config
                }
            }
            
        except Exception as e:
            logger.error(f"❌ RAPTOR tree building failed: {e}")
            raise
    
    async def _load_chunks_with_embeddings(
        self,
        doc_id: str,
        tenant_id: str,
        kb_id: str
    ) -> List[Tuple[str, List[float]]]:
        """Load existing chunks WITH their embeddings"""
        
        try:
            async with get_repositories() as repos:
                # Get all chunks for this document
                chunks = await repos.chunk_repo.get_chunks_by_document(doc_id, order_by_index=True)
                
                if not chunks:
                    return []
                
                # Get embeddings for these chunks (batch query for efficiency)
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                embeddings_dict = await repos.embedding_repo.get_embeddings_by_owners(
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    owner_type=EmbeddingOwnerType.chunk,
                    owner_ids=chunk_ids
                )
                
                chunks_with_embeddings = []
                for chunk in chunks:
                    embedding = embeddings_dict.get(chunk.chunk_id)
                    
                    if embedding and embedding.vector is not None and len(embedding.vector) > 0:
                        chunks_with_embeddings.append((chunk.content, embedding.vector))
                    else:
                        logger.warning(f"No embedding found for chunk {chunk.chunk_id}")
                
                logger.info(f"📦 Loaded {len(chunks_with_embeddings)} chunks with embeddings from {len(chunks)} total chunks")
                
                return chunks_with_embeddings
                
        except Exception as e:
            logger.error(f"Failed to load chunks with embeddings: {e}")
            raise
    
    async def _save_summary_chunks(
        self,
        augmented_chunks: List[Tuple[str, List[float], int]],  # Added level information
        original_count: int,
        doc_id: str,
        tenant_id: str,
        kb_id: str
    ) -> None:
        """Save ONLY RAPTOR summary chunks + embeddings (OPTIMIZED - no duplicate queries)"""
        
        try:
            async with get_repositories() as repos:
                
                # Extract ONLY summary chunks (everything after original count)
                summary_chunks = augmented_chunks[original_count:]
                
                if not summary_chunks:
                    logger.info("No RAPTOR summary chunks to save")
                    return
                
                logger.info(f"💾 Saving {len(summary_chunks)} RAPTOR summary chunks...")
                
                # Prepare bulk data for summary chunks + embeddings (memory-efficient batching)
                chunk_data_list = []
                embedding_data_list = []
                
                logger.info(f"💾 Processing {len(summary_chunks)} summary chunks in memory...")
                
                for i, (content, embedding, level) in enumerate(summary_chunks):
                    chunk_id = f"raptor_{doc_id}_{i}"
                    
                    # Summary chunk data
                    chunk_info = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "content": content,
                        "chunk_index": original_count + i,  # Continue from original chunks
                        "token_count": int(len(content.split()) * 1.3),  # Rough estimate
                        "meta": {
                            "type": "raptor_summary",
                            "level": level,  # Use actual level from build_tree
                            "created_by": "raptor_builder"
                        }
                    }
                    chunk_data_list.append(chunk_info)
                    
                    # Summary embedding data
                    embedding_info = {
                        "id": f"{EmbeddingOwnerType.summary.value}::{chunk_id}",
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "owner_id": chunk_id,
                        "owner_type": EmbeddingOwnerType.summary,  # Use summary type for RAPTOR summaries
                        "vector": embedding,
                        "model": self.embed_config.embed_model,
                        "dimension": len(embedding),
                        "meta": {
                            "doc_id": doc_id,
                            "type": "raptor_summary",
                            "level": level  # Use actual level from build_tree
                        }
                    }
                    embedding_data_list.append(embedding_info)
                
                # Bulk save summary chunks
                await repos.chunk_repo.bulk_upsert_chunks(chunk_data_list)
                
                # Bulk save summary embeddings
                await repos.embedding_repo.bulk_upsert_embeddings(embedding_data_list)
                
                # Update KB statistics using atomic increment (race-condition safe)
                summary_tokens = sum(chunk_info['token_count'] for chunk_info in chunk_data_list)
                await repos.kb_repo.increment_kb_stats(
                    kb_id=kb_id,
                    chunk_count_delta=len(summary_chunks),  # Add RAPTOR summary chunks
                    total_tokens_delta=summary_tokens  # Add RAPTOR summary tokens
                )
                logger.info(f"🔢 KB stats incremented: +{len(summary_chunks)} summary chunks, +{summary_tokens} summary tokens")
                
                # Commit all changes
                await repos.commit()
                
                logger.info(f"✅ RAPTOR summaries saved: {len(summary_chunks)} chunks + embeddings")
                
        except Exception as e:
            logger.error(f"Failed to save summary chunks: {e}")
            raise
    
    async def get_raptor_stats(self, doc_id: str) -> Dict[str, Any]:
        """Get RAPTOR tree statistics for a document"""
        
        try:
            async with get_repositories() as repos:
                # Get all chunks for this document
                all_chunks = await repos.chunk_repo.get_chunks_by_doc_id(doc_id)
                
                # Separate original chunks from summaries
                original_chunks = [c for c in all_chunks if c.metadata.get('type') != 'raptor_summary']
                summary_chunks = [c for c in all_chunks if c.metadata.get('type') == 'raptor_summary']
                
                # Count by level
                level_counts = {}
                for chunk in summary_chunks:
                    level = chunk.metadata.get('level', 0)
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                return {
                    "doc_id": doc_id,
                    "original_chunks": len(original_chunks),
                    "summary_chunks": len(summary_chunks),
                    "total_chunks": len(all_chunks),
                    "tree_levels": max(level_counts.keys()) if level_counts else 0,
                    "level_distribution": level_counts,
                    "raptor_enabled": len(summary_chunks) > 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get RAPTOR stats: {e}")
            return {"error": str(e)}
