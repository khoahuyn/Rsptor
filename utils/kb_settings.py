from typing import Dict, Any


def get_default_kb_settings() -> Dict[str, Any]:
    """Get default settings for new knowledge bases from config"""
    from config.chunking import get_chunking_settings
    from config.embedding import get_embedding_settings
    from config.raptor import get_raptor_settings
    
    # Get current configs
    chunk_config = get_chunking_settings()
    embed_config = get_embedding_settings()
    raptor_config = get_raptor_settings()
    
    return {
        # Chunking settings from config
        "chunk_size": chunk_config.chunk_size,
        "chunk_overlap_percent": chunk_config.chunk_overlap_percent,
        "hierarchical_pattern_set": chunk_config.hierarchical_pattern_set,
        
        # Embedding settings from config 
        "embedding_model": embed_config.embed_model,
        "vector_dimension": embed_config.embed_dimension,
        
        # RAGFlow/RAPTOR settings from config
        "enable_raptor": True,
        "max_levels": raptor_config.max_levels,
        "max_clusters": raptor_config.max_clusters,
        "similarity_threshold": raptor_config.similarity_threshold,
        
        # Retrieval settings (defaults)
        "top_k": 8
    }
