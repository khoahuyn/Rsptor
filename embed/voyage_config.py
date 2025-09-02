from typing import Dict, Any


class VoyageConfig:
    """Configuration constants for VoyageAI service"""
    
    # Rate limiting (actual Voyage free tier limits)
    DEFAULT_RPM_LIMIT: int = 2        # Under 3 RPM limit (safety margin) ✅
    DEFAULT_TPM_LIMIT: int = 8000     # Under 10K TPM limit (safety margin) ✅
    DEFAULT_MAX_RETRIES: int = 3      
    
    
    
    # Adaptive batch processing (optimized for multi-key setup)
    DEFAULT_BATCH_SIZE: int = 8       
    MIN_BATCH_SIZE: int = 2            
    MAX_BATCH_SIZE: int = 32          
    
    BATCH_DELAY_SECONDS: float = 45.0 
    
    MIN_EMBED_INTERVAL: float = 0.5   
    
    @classmethod
    def get_optimized_config(cls, is_production: bool = False) -> Dict[str, Any]:
        # SIMPLIFIED: Always return basic config since we use 4 keys instead of production tier
        return {
            "rpm_limit": cls.DEFAULT_RPM_LIMIT,
            "tpm_limit": cls.DEFAULT_TPM_LIMIT,
            "max_retries": cls.DEFAULT_MAX_RETRIES,
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "min_batch_size": cls.MIN_BATCH_SIZE,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "min_embed_interval": cls.MIN_EMBED_INTERVAL
        }
