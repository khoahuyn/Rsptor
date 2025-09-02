import logging
import time
from typing import Dict, Optional

from models import RetrievalRequest, RetrievalResponse, RetrievedNode, RetrievalStats
from config.embedding import get_embedding_settings
from config.retrieval import get_retrieval_config
from config.cache import get_cache_settings
from config.raptor import get_raptor_settings
from embed.embedding import embed_texts
from database.repository_factory import get_repositories
from sqlalchemy import select
from models.database.embedding import EmbeddingORM

from .universal_query_enhancer import universal_query_enhancer
from .vector_index import get_vector_index
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
                logger.info("🎯 Cache hit!")
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
        
        # Simple cache cleanup
        if len(self.cache) > self.cache_config.retrieval_cache_max_entries:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    async def retrieve(self, req: RetrievalRequest) -> RetrievalResponse:
        """
        Enhanced RAGFlow retrieval with query enhancement and fast vector search
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Enhanced retrieval: '{req.query}' in KB {req.kb_id}")
            
            # Step 1: Check cache
            import hashlib
            query_hash = hashlib.md5(f"{req.query}_{req.tenant_id}_{req.kb_id}_{req.top_k}".encode()).hexdigest()
            cached_result = self._get_from_cache(query_hash)
            if cached_result:
                return RetrievalResponse(**cached_result)
            
            # Step 2: Enhance query (Universal RAGFlow approach)
            enhanced_query, keywords = universal_query_enhancer.enhance_query(req.query)
            query_tokens = enhanced_query.lower().split()
            
            # Step 3: Generate embeddings
            query_vectors = await embed_texts(
                texts=[enhanced_query],
                vector_dim=self.embed_config.embed_dimension,
                cfg=self.embed_config
            )
            query_vector = query_vectors[0]
            
            async with get_repositories() as repos:
                # Step 4: Build/use vector index for fast search
                index_ready = await build_vector_index(req.tenant_id, req.kb_id, repos, self.embed_config)

                logger.info(f"🔧 Vector index ready: {index_ready}")
                
                # Enable FAISS vector search for better performance
                if index_ready:
                    # Fast vector search with FAISS
                    index = get_vector_index(req.kb_id, self.embed_config.embed_dimension)
                    logger.info(f"📊 Vector index stats: {index.get_stats()}")
                    
                    # RAGFlow approach: Get large candidate pool for reranking  
                    total_candidates = req.top_k * req.candidate_multiplier
                    vector_results = await index.search_async(query_vector, total_candidates)

                    logger.info(f"⚡ Fast vector search: {len(vector_results)} candidates from index of {index.size()} vectors")
                    
                    # Convert FAISS results to scored chunks
                    scored_chunks = await convert_vector_results_to_chunks(
                        vector_results, req, repos, keywords, query_vector
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
                    
                    scored_chunks = []
                    
                    # Process all embeddings for database search
                    for emb in all_embeddings:
                        if emb.vector is None or len(emb.vector) == 0:
                            continue
                            
                        try:
                            # Get chunk content - fix transaction issue
                            chunk = await repos.chunk_repo.get_by_id(emb.owner_id)
                            if not chunk:
                                logger.warning(f"Chunk not found: {emb.owner_id}")
                                continue
                            
                            # Calculate advanced similarities
                            similarities = calculate_advanced_similarity(
                                keywords,
                                chunk.content,
                                query_vector,
                                emb.vector,
                                {'chunk_index': chunk.chunk_index, 'owner_type': emb.owner_type.value}
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
                
                # Step 5: Apply threshold and sort
                threshold = req.min_similarity_threshold
                valid_chunks = [c for c in scored_chunks if c['final_score'] >= threshold]
                
                # Fallback strategy (reuse raptor similarity_threshold)
                if not valid_chunks and threshold > self.raptor_config.similarity_threshold:
                    fallback_threshold = max(0.05, threshold * 0.5)  # Auto calculation
                    valid_chunks = [c for c in scored_chunks if c['final_score'] >= fallback_threshold]
                    logger.info(f"🔄 Fallback applied: {len(valid_chunks)} chunks found")
                
                # Sort by final score
                valid_chunks.sort(key=lambda x: x['final_score'], reverse=True)
                
                # Step 6: Create response
                retrieved_nodes = []
                total_tokens = 0
                
                for chunk in valid_chunks[:req.top_k]:
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
                            'enhanced_query': enhanced_query,
                            'keywords': keywords,
                            **chunk['similarities']  # Include all similarity metrics
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
                    filtered_candidates=len(valid_chunks),
                    search_method="enhanced_ragflow_hybrid",
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
                
                logger.info(f"✅ Enhanced retrieval completed in {processing_time:.2f}s: {len(retrieved_nodes)} nodes")
                
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
