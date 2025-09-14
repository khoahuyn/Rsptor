# Retrieval Services
# Enhanced RAGFlow-inspired retrieval with universal query enhancement

# Main retrieval service (refactored from enhanced_retrieval.py)
from .core import enhanced_ragflow_retrieval, EnhancedRAGFlowRetrieval

# Helper functions and data structures
from .retrieval_helper import (
    EnhancedSearchResult,
    calculate_advanced_similarity,
    calculate_final_score,
    build_vector_index,
    convert_vector_results_to_chunks
)

# Supporting services
from .universal_query_enhancer import universal_query_enhancer  
from .persistent_vector_index import get_persistent_vector_index, clear_persistent_vector_index

# Configuration
from config.retrieval import get_retrieval_config, set_retrieval_config, RetrievalConfig

__all__ = [
    # Main retrieval service (backward compatibility)
    "enhanced_ragflow_retrieval",
    "EnhancedRAGFlowRetrieval",
    
    
    # Helper functions
    "EnhancedSearchResult",
    "calculate_advanced_similarity", 
    "calculate_final_score",
    "build_vector_index",
    "convert_vector_results_to_chunks",
    
    # Query enhancement
    "universal_query_enhancer",
    
    # Vector indexing  
    "get_persistent_vector_index",
    "clear_persistent_vector_index",
    
    # Configuration
    "get_retrieval_config",
    "set_retrieval_config", 
    "RetrievalConfig",
]

