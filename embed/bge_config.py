from typing import Dict, Any
from .embedding_constants import EmbeddingConstants


class BGEConfig:
    """Configuration constants for BGE-M3 local service"""
    
    # Rate limiting (optimized for stable local performance)
    DEFAULT_RPM_LIMIT: int = 60        
    DEFAULT_MAX_RETRIES: int = EmbeddingConstants.DEFAULT_RETRY_ATTEMPTS  
    
    # Batch processing (adaptive for optimal performance) 
    DEFAULT_BATCH_SIZE: int = 12       
    MIN_BATCH_SIZE: int = 4            
    MAX_BATCH_SIZE: int = 24           
    
    # API timeouts (optimized for local)
    REQUEST_TIMEOUT_SECONDS: int = EmbeddingConstants.DEFAULT_TIMEOUT  
    
    MIN_EMBED_INTERVAL: float = 0.05  # Fast for local BGE (no rate limits)
    
    
    @classmethod
    def get_optimized_config(cls) -> Dict[str, Any]:
        """Get optimized config for local BGE (adaptive_batch.py will use defaults for missing values)"""
        return {
            "rpm_limit": cls.DEFAULT_RPM_LIMIT,
            "max_retries": cls.DEFAULT_MAX_RETRIES,
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "min_batch_size": cls.MIN_BATCH_SIZE,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "timeout": cls.REQUEST_TIMEOUT_SECONDS,
            "min_embed_interval": cls.MIN_EMBED_INTERVAL
        }
