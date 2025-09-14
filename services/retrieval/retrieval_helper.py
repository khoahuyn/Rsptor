import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sqlalchemy import select

from config.retrieval import get_retrieval_config
from models.database.embedding import EmbeddingORM
from models.database.document import ChunkORM
from .universal_query_enhancer import universal_query_enhancer


logger = logging.getLogger("retrieval_helper")


def _calculate_quick_score(vector_similarity: float, content: str, chunk_meta: Dict, config) -> Dict[str, float]:
    """Quick scoring for early termination (RAGFlow-style simple approach)"""
    return {
        'text_similarity': 0.0,  # Skip expensive text similarity
        'vector_similarity': vector_similarity
    }


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with more metadata"""
    total: int
    chunks: List[Dict]
    query_vector: List[float]
    enhanced_query: str
    keywords: List[str]
    processing_time: float


def calculate_advanced_similarity(
    query_tokens: List[str],
    content: str,
    query_vector: List[float],
    doc_vector: List[float],
    chunk_meta: Dict,
    vector_similarity: float = None
) -> Dict[str, float]:
    """Calculate multiple similarity metrics - Universal RAGFlow approach"""
    config = get_retrieval_config()
    
    # 1. Universal text similarity (no hardcoded terms)
    content_tokens = content.lower().split()
    text_similarity = universal_query_enhancer.calculate_text_similarity(query_tokens, content_tokens)
    
    # 2. Vector cosine similarity  
    if vector_similarity is None:
    # Fallback calculation (for non-FAISS path)
        try:
            q_vec = np.array(query_vector)
            d_vec = np.array(doc_vector)
        
            if len(q_vec) == len(d_vec) and np.linalg.norm(q_vec) > 0 and np.linalg.norm(d_vec) > 0:
                vector_similarity = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
            else:
                vector_similarity = 0.0
        except:
            vector_similarity = 0.0
    
    
    # Early termination for low-similarity chunks (skip expensive calculations)
    if vector_similarity is not None and vector_similarity < config.early_exit_threshold:
        return _calculate_quick_score(vector_similarity, content, chunk_meta, config)
    
    # RAGFlow-style simple scoring: only text + vector similarity
    return {
        'text_similarity': text_similarity,
        'vector_similarity': vector_similarity
    }


def calculate_final_score(similarities: Dict[str, float], chunk_meta: Dict) -> float:
    """RAGFlow-style simple final score calculation"""
    config = get_retrieval_config()
    
    # Simple hybrid score: text_weight = 1.0 - vector_weight
    text_weight = 1.0 - config.vector_similarity_weight
    final_score = (similarities['text_similarity'] * text_weight + 
                   similarities['vector_similarity'] * config.vector_similarity_weight)
    
    return min(final_score, 1.0)  # Cap at 1.0


async def build_vector_index(tenant_id: str, kb_id: str, repos, embed_config) -> bool:
    """Build or load persistent vector index for fast search"""
    try:
        from .persistent_vector_index import create_persistent_index, persistent_indexes
        
        # Create persistent index instead of regular one
        persistent_index = create_persistent_index(kb_id, embed_config.embed_dimension)
        
        # Get current embedding count from DB first
        stmt = select(EmbeddingORM).where(
            EmbeddingORM.tenant_id == tenant_id,
            EmbeddingORM.kb_id == kb_id
        )
        
        result = await repos.session.execute(stmt)
        embeddings = result.scalars().all()
        current_count = len(embeddings)
        
        logger.info(f"🔍 Found {current_count} embeddings in DB")
        
        # Try to load existing persistent index
        if persistent_index.load_from_disk():
            # Check if index is still valid
            if not persistent_index.is_index_stale(current_count):
                logger.info(f"📁 Using persistent index: {len(persistent_index.chunk_ids)} vectors")
                
                # Register in global persistent_indexes dict
                key = f"{kb_id}_{embed_config.embed_dimension}"
                persistent_indexes[key] = persistent_index
                
                return True
            else:
                logger.info(f"🔄 Persistent index stale, rebuilding...")
        else:
            logger.info(f"🆕 No persistent index found, building new one...")
        
        logger.info(f"🔧 Building persistent vector index for KB {kb_id}...")
        
        logger.info(f"🔍 Processing {len(embeddings)} embeddings for index building")
        
        vectors = []
        chunk_ids = []
        metadata = []
        
        for i, emb in enumerate(embeddings):
            if emb.vector is None:
                logger.debug(f"⚠️ Embedding {i}: vector is None")
                continue
            if len(emb.vector) == 0:
                logger.debug(f"⚠️ Embedding {i}: vector is empty")
                continue
                
            vectors.append(emb.vector)
            chunk_ids.append(emb.owner_id)
            metadata.append({
                'owner_type': emb.owner_type.value,
                'embedding_model': emb.model,
                'doc_id': emb.meta.get('doc_id') if emb.meta else None
            })
            
            if i < 3:  # Log first few for debugging
                logger.debug(f"✅ Added vector {i}: {emb.owner_type.value} {emb.owner_id}, dim={len(emb.vector)}")
        
        logger.info(f"📦 Prepared {len(vectors)} valid vectors for persistent index")
        
        if vectors:
            persistent_index.add_vectors(vectors, chunk_ids, metadata)
            logger.info(f"✅ Vector index built: {len(vectors)} vectors")
            
            # Save to disk for persistence
            if persistent_index.save_to_disk():
                logger.info(f"💾 Persistent index saved to disk")
            else:
                logger.warning(f"⚠️ Failed to save persistent index to disk")
            
            # Register in global persistent_indexes dict
            key = f"{kb_id}_{embed_config.embed_dimension}"
            persistent_indexes[key] = persistent_index
            
            return True
        else:
            logger.warning("❌ No vectors found to build index - all embeddings invalid")
            return False
            
    except Exception as e:
        logger.error(f"Error building persistent vector index: {e}")
        return False


async def convert_vector_results_to_chunks(
    vector_results: List[Tuple[str, float, Dict]], 
    req,
    repos,
    query_tokens: List[str], 
    query_vector: List[float]
) -> List[Dict]:
    """
    Convert FAISS vector search results to scored chunks format
    Handles all 3 owner_types: chunk, summary, root
    OPTIMIZED: Batch load all chunks in single DB query + query preprocessing
    """
    if not vector_results:
        return []
    
    # RAGFlow-style: No query preprocessing needed for simplified scoring
    config = get_retrieval_config()
    logger.debug("🚀 Using simplified RAGFlow-style scoring (no preprocessing needed)")
        
    # Step 1: Batch load all chunks in single query (60 chunks → 1 DB call!)
    chunk_ids = [chunk_id for chunk_id, _, _ in vector_results]

    # Batch query all chunks at once
    stmt = select(ChunkORM).where(ChunkORM.chunk_id.in_(chunk_ids))
    result = await repos.session.execute(stmt)
    chunks_list = result.scalars().all()
    
    # Create lookup dictionary for O(1) access
    chunks_dict = {chunk.chunk_id: chunk for chunk in chunks_list}

    scored_chunks = []
    
    # Step 2: Process all results using cached chunks
    for chunk_id, vector_similarity, metadata in vector_results:
        try:
            # Get chunk content from cached dictionary (O(1) lookup!)
            chunk = chunks_dict.get(chunk_id)
            if not chunk:
                logger.warning(f"Chunk not found in batch: {chunk_id}")
                continue
            
            # Get owner_type from metadata
            owner_type = metadata.get('owner_type', 'chunk')
            
            # Calculate advanced similarities (RAGFlow-style simple approach)
            similarities = calculate_advanced_similarity(
                query_tokens,
                chunk.content,
                query_vector,
                [],  # Don't need doc_vector as we already have vector_similarity
                {
                    'chunk_index': chunk.chunk_index, 
                    'owner_type': owner_type
                },
                vector_similarity=vector_similarity
            )
            
            # Override vector similarity from FAISS (more accurate)
            similarities['vector_similarity'] = vector_similarity
            
            # Calculate final score with owner_type bonuses
            final_score = calculate_final_score(similarities, {
                'owner_type': owner_type,
                'chunk_index': chunk.chunk_index
            })
            
            scored_chunks.append({
                'chunk_id': chunk_id,
                'content': chunk.content,
                'final_score': final_score,
                'similarities': similarities,
                'owner_type': owner_type,
                'embedding_model': metadata.get('embedding_model', ''),
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'token_count': chunk.token_count
            })
            
            logger.debug(f"✅ Converted {owner_type} chunk {chunk_id}: score={final_score:.3f}")
            
        except Exception as e:
            logger.warning(f"Error converting vector result {chunk_id}: {e}")
            # Rollback transaction on error
            try:
                await repos.session.rollback()
            except:
                pass
            continue
            
    logger.info(f"🔄 FAISS results converted: {len(scored_chunks)} valid chunks")
    return scored_chunks
