import re
from typing import List
from functools import lru_cache
from models.document import DocumentChunk
from utils import token_count


class ChunkOptimizer:
    """Utility class for chunking performance optimizations"""
    
    @staticmethod
    def setup_pattern_cache(bullet_patterns: List[str], pattern_cache_enabled: bool) -> List[re.Pattern]:
        """
        Precompile regex patterns for performance
        
        Args:
            bullet_patterns: List of regex patterns
            pattern_cache_enabled: Whether to enable pattern caching
            
        Returns:
            List of compiled regex patterns
        """
        if pattern_cache_enabled:
            return [re.compile(pattern) for pattern in bullet_patterns]
        return []

    @staticmethod
    def get_text_by_token_count(text: str, target_tokens: int) -> str:
        """
        Extract text portion that contains approximately target_tokens
        RAGFlow-style token-aware text extraction for overlap using REAL token counting
        
        Args:
            text: Input text
            target_tokens: Target number of tokens
            
        Returns:
            Text portion with approximately target_tokens
        """
        if target_tokens <= 0:
            return ""
        
        # Split by words for precision
        words = text.split()
        if not words:
            return ""
        
        # Start from the end and work backwards (for overlap)
        # Use REAL token counting instead of estimation
        current_tokens = 0
        selected_words = []
        
        for word in reversed(words):
            # Test adding this word
            test_text = " ".join([word] + selected_words)
            test_tokens = token_count(test_text)
            
            if test_tokens > target_tokens:
                break
                
            selected_words.insert(0, word)  # Add to beginning
            current_tokens = test_tokens
        
        return " ".join(selected_words)

    @staticmethod
    def batch_token_count(texts: List[str], token_cache_enabled: bool, 
                         cached_token_count_func=None) -> List[int]:
        """
        Batch token counting for performance
        Process multiple texts at once to reduce overhead
        
        Args:
            texts: List of texts to count tokens for
            token_cache_enabled: Whether token caching is enabled
            cached_token_count_func: Cached token count function
            
        Returns:
            List of token counts
        """
        if not texts:
            return []
        
        # If token caching is enabled, still count individually (cache benefits)
        if token_cache_enabled and cached_token_count_func:
            return [cached_token_count_func(text) for text in texts]
        
        # Otherwise, fallback to individual counting
        return [token_count(text) for text in texts]


class ChunkStatistics:
    """Utility class for chunk statistics generation"""
    
    @staticmethod
    def get_chunking_stats(chunks: List[DocumentChunk], chunk_size: int, 
                          pattern_set: int, overlap_percent: int) -> dict:
        """
        Generate hierarchical chunking statistics
        
        Args:
            chunks: List of DocumentChunk objects
            chunk_size: Configured chunk size
            pattern_set: Pattern set used
            overlap_percent: Overlap percentage
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "chunking_method": "hierarchical_structure_aware_optimized",
                "chunk_size": chunk_size,
                "structure_detected": False
            }
        
        return {
            "total_chunks": len(chunks),
            "chunking_method": "hierarchical_structure_aware_optimized", 
            "chunk_size": chunk_size,
            "pattern_set": pattern_set,
            "structure_detected": True,
            "overlap_percent": overlap_percent,
            "performance_optimizations": True
        }


class TokenCacheManager:
    """Manager for token counting cache functionality"""
    
    @staticmethod
    def create_cached_token_counter(cache_enabled: bool, cache_max_size: int = 10000):
        """
        Create cached token counter function
        
        Args:
            cache_enabled: Whether to enable caching
            cache_max_size: Maximum cache size
            
        Returns:
            Token counting function (cached or regular)
        """
        if cache_enabled:
            return lru_cache(maxsize=cache_max_size)(token_count)
        return token_count
