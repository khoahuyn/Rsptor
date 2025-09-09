import logging
import time
from typing import Dict, Optional, Any, List

from models import RetrievalRequest, RetrievalResponse, RetrievedNode, RetrievalStats
from config.embedding import get_embedding_settings
from config.retrieval import get_retrieval_config
from config.cache import get_cache_settings
from config.raptor import get_raptor_settings
from embed.embedding import embed_texts
from database.repository_factory import get_repositories
from sqlalchemy import select
from models.database.embedding import EmbeddingORM
from models.database.document import ChunkORM

from .universal_query_enhancer import universal_query_enhancer
from .persistent_vector_index import get_persistent_vector_index
from .retrieval_helper import (
    build_vector_index,
    convert_vector_results_to_chunks,
    calculate_advanced_similarity,
    calculate_final_score
)

logger = logging.getLogger("enhanced_retrieval_core")


class EnhancedRAGFlowRetrieval:
    """Enhanced RAGFlow retrieval with FAISS vector search and hybrid scoring"""

    def __init__(self):
        self.embed_config = get_embedding_settings()
        self.retrieval_config = get_retrieval_config()
        self.cache_config = get_cache_settings()
        self.raptor_config = get_raptor_settings()
        self.cache = {}  # Simple query cache
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry.get('timestamp', 0) < self.cache_config.retrieval_cache_ttl_seconds
    
    def _get_from_cache(self, query_hash: str) -> Optional[Dict]:
        """Get result from cache if valid"""
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if self._is_cache_valid(entry):
                # 🚀 PERFORMANCE: Track cache hits for smart cleanup
                entry['hit_count'] = entry.get('hit_count', 0) + 1
                entry['last_accessed'] = time.time()
                logger.info(f"🎯 Cache hit! (used {entry['hit_count']} times)")
                return entry['result']
            else:
                del self.cache[query_hash]
        return None
    
    def _save_to_cache(self, query_hash: str, result: Dict):
        """Save result to cache"""
        self.cache[query_hash] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # 🚀 PERFORMANCE: Smarter cache cleanup with LRU + hit frequency
        if len(self.cache) > self.cache_config.retrieval_cache_max_entries:
            # Remove cache entries with low hit frequency first
            cache_items = [(k, v) for k, v in self.cache.items()]
            cache_items.sort(key=lambda x: (x[1].get('hit_count', 0), x[1]['timestamp']))  # Low hits + old timestamp
            oldest_key = cache_items[0][0]
            del self.cache[oldest_key]
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        if not self.cache:
            return {"size": 0, "hit_rate": 0.0, "avg_hits": 0.0}
        
        total_hits = sum(entry.get('hit_count', 0) for entry in self.cache.values())
        avg_hits = total_hits / len(self.cache) if self.cache else 0
        
        return {
            "size": len(self.cache),
            "total_entries": len(self.cache),
            "total_hits": total_hits,
            "avg_hits_per_entry": round(avg_hits, 2),
            "most_hit": max((entry.get('hit_count', 0) for entry in self.cache.values()), default=0)
        }
    
    async def retrieve(self, req: RetrievalRequest) -> RetrievalResponse:
        """
        Enhanced RAGFlow retrieval with FAISS vector search and hybrid scoring
        """
        try:
            logger.info(f"🚀 Enhanced retrieval: '{req.query}' in KB {req.kb_id}")
            return await self._standard_retrieval(req)
                
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            raise
    
    async def _standard_retrieval(self, req: RetrievalRequest) -> RetrievalResponse:
        """
        Standard enhanced RAGFlow retrieval with query enhancement and fast vector search
        """
        start_time = time.time()
        timings = {}  # Track performance of each step
        
        try:
            logger.info(f"🚀 Enhanced retrieval: '{req.query}' in KB {req.kb_id}")
            
            # Step 1: Check cache
            import hashlib
            # 🚀 PERFORMANCE: Normalize query for better cache hits
            normalized_query = req.query.lower().strip()
            # Remove extra spaces and normalize punctuation
            normalized_query = ' '.join(normalized_query.split())
            query_hash = hashlib.md5(f"{normalized_query}_{req.tenant_id}_{req.kb_id}_{req.top_k}".encode()).hexdigest()
            cached_result = self._get_from_cache(query_hash)
            if cached_result:
                # 🚀 PERFORMANCE: Log cache effectiveness
                cache_stats = self._get_cache_stats()
                logger.info(f"📊 Cache stats: {cache_stats}")
                return RetrievalResponse(**cached_result)
            
            # Step 2: Enhance query (Universal RAGFlow approach)
            t2 = time.time()
            enhanced_query, keywords = universal_query_enhancer.enhance_query(req.query)
            query_tokens = enhanced_query.lower().split()
            timings["query_enhancement"] = (time.time() - t2) * 1000
            
            
            # Step 3: Generate embeddings
            t3 = time.time()
            query_vectors = await embed_texts(
                texts=[enhanced_query],
                vector_dim=self.embed_config.embed_dimension,
                cfg=self.embed_config
            )
            query_vector = query_vectors[0]
            timings["embedding_generation"] = (time.time() - t3) * 1000
            
            async with get_repositories() as repos:
                # Step 4: Build/use vector index for fast search
                index_ready = await build_vector_index(req.tenant_id, req.kb_id, repos, self.embed_config)

                logger.info(f"🔧 Vector index ready: {index_ready}")
                
                # Enable FAISS vector search for better performance
                if index_ready:
                    # Fast vector search with FAISS (using persistent index)
                    index = get_persistent_vector_index(req.kb_id, self.embed_config.embed_dimension)
                    logger.info(f"📊 Persistent vector index stats: {index.get_stats()}")
                    
                    # RAGFlow approach: Get large candidate pool for reranking  
                    total_candidates = req.top_k * req.candidate_multiplier
                    vector_results = await index.search_async(query_vector, total_candidates)

                    logger.info(f"⚡ Fast vector search: {len(vector_results)} candidates from index of {index.size()} vectors")
                    
                    # Convert FAISS results to scored chunks
                    scored_chunks = await convert_vector_results_to_chunks(
                        vector_results, req, repos, query_tokens, query_vector
                    )

                else:
                    # Fallback to database search with enhanced scoring
                    logger.info("💾 Using database search fallback")
                    
                    # Get all embeddings for hybrid search
                    stmt = select(EmbeddingORM).where(
                        EmbeddingORM.tenant_id == req.tenant_id,
                        EmbeddingORM.kb_id == req.kb_id
                    )
                    
                    result = await repos.session.execute(stmt)
                    all_embeddings = result.scalars().all()
                    
                    logger.info(f"📊 Found {len(all_embeddings)} embeddings for database search")
                    
                    # Preprocess query ONCE for all embeddings (optimization!)
                    from config.retrieval import get_retrieval_config
                    from .retrieval_helper import preprocess_query_for_scoring
                    config = get_retrieval_config()
                    preprocessed_query = preprocess_query_for_scoring(query_tokens, config)
                    logger.debug(f"🚀 Query preprocessed for DB path: {len(preprocessed_query['valid_tokens'])} valid tokens")
                    
                    scored_chunks = []
                    
                    # OPTIMIZATION: Batch load all chunks to avoid N+1 queries
                    embedding_chunk_ids = [emb.owner_id for emb in all_embeddings if emb.vector is not None and len(emb.vector) > 0]
                    
                    # Single bulk query instead of N individual queries
                    chunk_stmt = select(ChunkORM).where(ChunkORM.chunk_id.in_(embedding_chunk_ids))
                    chunk_result = await repos.session.execute(chunk_stmt)
                    chunks_list = chunk_result.scalars().all()
                    
                    # Create lookup dictionary for O(1) access
                    chunks_dict = {chunk.chunk_id: chunk for chunk in chunks_list}
                    logger.info(f"🚀 Batch loaded {len(chunks_list)} chunks in single query (avoiding {len(embedding_chunk_ids)} N+1 queries)")
                    
                    # Process all embeddings using cached chunks
                    for emb in all_embeddings:
                        if emb.vector is None or len(emb.vector) == 0:
                            continue
                            
                        try:
                            # Get chunk from cache instead of DB query
                            chunk = chunks_dict.get(emb.owner_id)
                            if not chunk:
                                logger.warning(f"Chunk not found in batch: {emb.owner_id}")
                                continue
                            
                            # Calculate advanced similarities (optimized with preprocessed query)
                            similarities = calculate_advanced_similarity(
                                query_tokens,  # 🎯 Use enhanced query tokens, not keywords
                                chunk.content,
                                query_vector,
                                emb.vector,
                                {'chunk_index': chunk.chunk_index, 'owner_type': emb.owner_type.value},
                                vector_similarity=None,
                                preprocessed_query=preprocessed_query  # ← OPTIMIZATION: Use preprocessed query!
                            )
                            
                            final_score = calculate_final_score(similarities, {
                                'owner_type': emb.owner_type.value,
                                'chunk_index': chunk.chunk_index
                            })
                            
                            scored_chunks.append({
                                'chunk_id': emb.owner_id,
                                'content': chunk.content,
                                'final_score': final_score,
                                'similarities': similarities,
                                'owner_type': emb.owner_type.value,
                                'embedding_model': emb.model,
                                'doc_id': chunk.doc_id,
                                'chunk_index': chunk.chunk_index,
                                'token_count': chunk.token_count
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error processing embedding {emb.id}: {e}")
                            # Rollback transaction on error to prevent "InFailedSqlTransaction"
                            try:
                                await repos.session.rollback()
                            except:
                                pass
                            continue
                
                
                # Step 5: Simple top_k selection (core RAG behavior)
                sorted_chunks = sorted(scored_chunks, key=lambda x: x['final_score'], reverse=True)
                selected_chunks = sorted_chunks[:req.top_k]
                
                logger.info(f"🎯 Selected top {len(selected_chunks)} chunks from {len(scored_chunks)} candidates")
                
                # Step 6: Create response nodes
                retrieved_nodes = []
                total_tokens = 0
                
                for chunk in selected_chunks:
                    chunk_tokens = chunk.get(
                        'token_count', 
                        len(chunk['content'].split()) * self.retrieval_config.tokens_per_word
                    )
                    
                    # Check token budget
                    if req.token_budget and total_tokens + chunk_tokens > req.token_budget:
                        logger.info(f"💰 Token budget reached: {total_tokens}/{req.token_budget}")
                        break
                    
                    node = RetrievedNode(
                        node_id=chunk['chunk_id'],
                        similarity_score=chunk['final_score'],
                        content=chunk['content'],
                        level=0 if chunk['owner_type'] == 'chunk' else 1,
                        token_count=int(chunk_tokens),
                        meta={
                            'owner_type': chunk['owner_type'],
                            'embedding_model': chunk['embedding_model'],
                            'doc_id': chunk['doc_id'],
                            'chunk_index': chunk['chunk_index'],
                            # 'enhanced_query': enhanced_query,    
                            # 'keywords': keywords,                
                            'text_similarity': chunk['similarities']['text_similarity'],
                            'vector_similarity': chunk['similarities']['vector_similarity']
                            # 'keyword_bonus': chunk['similarities']['keyword_bonus'],    
                            # 'position_bonus': chunk['similarities']['position_bonus'],     
                            # 'length_bonus': chunk['similarities']['length_bonus']       
                        }
                    )
                    
                    retrieved_nodes.append(node)
                    total_tokens += chunk_tokens
                
                processing_time = time.time() - start_time
                
                # Create response stats (include cache performance)
                from utils import get_cache_stats
                cache_stats = get_cache_stats()
                
                stats = RetrievalStats(
                    query_tokens=len(query_tokens),
                    total_candidates=len(scored_chunks),
                    filtered_candidates=len(selected_chunks),
                    search_method="enhanced_ragflow_hybrid_simple",
                    embedding_model=self.embed_config.embed_model,
                    cache_hit_rate=cache_stats.get("embed_cache", {}).get("hit_rate", "0%"),
                    cache_size=cache_stats.get("embed_cache", {}).get("size", 0)
                )
                
                response = RetrievalResponse(
                    retrieved_nodes=retrieved_nodes,
                    retrieval_stats=stats
                )
                
                # Save to cache
                self._save_to_cache(query_hash, response.dict())
                
                logger.info(f"✅ Enhanced retrieval completed in {processing_time:.2f}s: {len(retrieved_nodes)} nodes (simple_top_k)")
                
                return response
                
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            
            # Return proper error response instead of fallback
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Retrieval failed",
                    "message": str(e),
                    "solution": "Please check server logs and try again"
                }
            )


# Global instance
enhanced_ragflow_retrieval = EnhancedRAGFlowRetrieval()
