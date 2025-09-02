import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy import select

from config.retrieval import get_retrieval_config
from models.database.embedding import EmbeddingORM, EmbeddingOwnerType
from models.database.document import ChunkORM
from .universal_query_enhancer import universal_query_enhancer
from .vector_index import get_vector_index

logger = logging.getLogger("retrieval_helper")


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
    chunk_meta: Dict
) -> Dict[str, float]:
    """Calculate multiple similarity metrics - Universal RAGFlow approach"""
    config = get_retrieval_config()
    
    # 1. Universal text similarity (no hardcoded terms)
    content_tokens = content.lower().split()
    text_similarity = universal_query_enhancer.calculate_text_similarity(query_tokens, content_tokens)
    
    # 2. Vector cosine similarity  
    try:
        q_vec = np.array(query_vector)
        d_vec = np.array(doc_vector)
        
        if len(q_vec) == len(d_vec) and np.linalg.norm(q_vec) > 0 and np.linalg.norm(d_vec) > 0:
            vector_similarity = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
        else:
            vector_similarity = 0.0
    except:
        vector_similarity = 0.0
    
    # 3. Universal keyword density (simplified approach)
    keyword_bonus = 0.0
    if query_tokens:
        content_lower = content.lower()
        matched_keywords = 0
        for token in query_tokens:
            if len(token) >= config.min_keyword_length and token in content_lower:
                # Simple weight based on token length
                weight = min(2.0, len(token) / 4.0)  # Max 2.0 weight
                keyword_bonus += 0.02 * weight  # Base 0.02 bonus
                matched_keywords += 1
        
        # Normalize and cap
        if matched_keywords > 0:
            keyword_bonus = min(config.max_keyword_bonus, keyword_bonus * (matched_keywords / len(query_tokens)))
    
    # 4. Position bonus (universal - early chunks often more important)
    position_bonus = 0.0
    chunk_index = chunk_meta.get('chunk_index', 0)
    if chunk_index < config.position_bonus_chunks:
        position_bonus = config.max_position_bonus * (config.position_bonus_chunks - chunk_index) / config.position_bonus_chunks
    
    # 5. Content length bonus (auto range: optimal ±75%)
    length_bonus = 0.0
    content_len = len(content)
    min_length = int(config.optimal_content_length * 0.25)  # 25% of optimal
    max_length = int(config.optimal_content_length * 2.5)   # 250% of optimal
    
    if min_length <= content_len <= max_length:
        # Peak bonus at optimal_length, taper off at extremes
        distance = abs(content_len - config.optimal_content_length)
        length_bonus = max(0, config.max_length_bonus * (1 - distance / config.optimal_content_length))
    
    return {
        'text_similarity': text_similarity,
        'vector_similarity': vector_similarity,
        'keyword_bonus': keyword_bonus,
        'position_bonus': position_bonus,
        'length_bonus': length_bonus
    }


def calculate_final_score(similarities: Dict[str, float], chunk_meta: Dict) -> float:
    """Calculate final ranking score with multiple factors"""
    config = get_retrieval_config()
    
    # Base hybrid score (auto text_weight = 1.0 - vector_weight)
    text_weight = 1.0 - config.vector_similarity_weight
    hybrid_score = (similarities['text_similarity'] * text_weight + 
                   similarities['vector_similarity'] * config.vector_similarity_weight)
    
    # Add bonuses
    final_score = (hybrid_score + 
                  similarities['keyword_bonus'] + 
                  similarities['position_bonus'] + 
                  similarities['length_bonus'])
    
    # Owner type boost
    owner_type = chunk_meta.get('owner_type', 'chunk')
    if owner_type == 'summary':
        final_score += config.summary_type_bonus
    elif owner_type == 'root':
        final_score += config.root_type_bonus
    
    return min(final_score, 1.0)  # Cap at 1.0


async def build_vector_index(tenant_id: str, kb_id: str, repos, embed_config) -> bool:
    """Build or update vector index for fast search"""
    try:
        index = get_vector_index(kb_id, embed_config.embed_dimension)
        
        # Check if index needs rebuilding
        existing_size = index.size()

        if existing_size > 0:
            logger.info(f"📦 Using existing vector index with {existing_size} vectors")
            return True
        else:
            logger.info(f"🆕 No existing index found (size={existing_size}), building new one...")
        
        logger.info(f"🔧 Building vector index for KB {kb_id}...")
        
        # Get all embeddings
        stmt = select(EmbeddingORM).where(
            EmbeddingORM.tenant_id == tenant_id,
            EmbeddingORM.kb_id == kb_id
        )
        
        result = await repos.session.execute(stmt)
        embeddings = result.scalars().all()
        
        logger.info(f"🔍 Found {len(embeddings)} embeddings in DB for index building")
        
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
        
        logger.info(f"📦 Prepared {len(vectors)} valid vectors for FAISS index")
        
        if vectors:
            index.add_vectors(vectors, chunk_ids, metadata)
            logger.info(f"✅ Vector index built: {len(vectors)} vectors")
            return True
        else:
            logger.warning("❌ No vectors found to build index - all embeddings invalid")
            return False
            
    except Exception as e:
        logger.error(f"Error building vector index: {e}")
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
    OPTIMIZED: Batch load all chunks in single DB query
    """
    if not vector_results:
        return []
        
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
            
            # Calculate advanced similarities (same as database path)
            similarities = calculate_advanced_similarity(
                query_tokens,
                chunk.content,
                query_vector,
                [],  # Don't need doc_vector as we already have vector_similarity
                {
                    'chunk_index': chunk.chunk_index, 
                    'owner_type': owner_type
                }
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
