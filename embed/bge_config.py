from typing import Dict, Any
from .embedding_constants import EmbeddingConstants


class BGEConfig:
    """Configuration constants for BGE-M3 local service"""
    
    # Local BGE optimized configuration (NO rate limiting needed)
    DEFAULT_MAX_RETRIES: int = EmbeddingConstants.DEFAULT_RETRY_ATTEMPTS  
    
    # Batch processing (AGGRESSIVE for optimal performance) 
    DEFAULT_BATCH_SIZE: int = 32       
    
    # API timeouts (GENEROUS for local)
    REQUEST_TIMEOUT_SECONDS: int = 60  
    
    
    @classmethod
    def get_optimized_config(cls) -> Dict[str, Any]:
        """Get optimized config for local BGE - AGGRESSIVE parallel processing like VoyageAI"""
        return {
            # ✅ NO rate limiting needed for local BGE
            "max_retries": cls.DEFAULT_MAX_RETRIES,
            "batch_size": cls.DEFAULT_BATCH_SIZE,    
            "timeout": cls.REQUEST_TIMEOUT_SECONDS,  
            "enable_parallel": True,                 
            "max_concurrent": 8                      
        }
