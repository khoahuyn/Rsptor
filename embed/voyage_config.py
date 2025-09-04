from typing import Dict, Any


class VoyageConfig:
    """Configuration constants for VoyageAI service"""
    
    # Rate limiting (Voyage free tier maximum limits)
    DEFAULT_RPM_LIMIT: int = 3        
    DEFAULT_TPM_LIMIT: int = 10000    
    DEFAULT_MAX_RETRIES: int = 5      # Increased for better rate limit resilience      
    
    
    # Parallel execution settings (conservative for stability)
    MIN_EMBED_INTERVAL: float = 21.0  # Conservative: 3 RPM = 20s + 1s buffer
    DEFAULT_CONCURRENT_PER_KEY: int = 2  # Conservative: Proven ZERO rate limits, excellent performance 
    
    @classmethod
    def get_optimized_config(cls, is_production: bool = False) -> Dict[str, Any]:
        return {
            "rpm_limit": cls.DEFAULT_RPM_LIMIT,
            "tpm_limit": cls.DEFAULT_TPM_LIMIT,
            "max_retries": cls.DEFAULT_MAX_RETRIES,
            "min_embed_interval": cls.MIN_EMBED_INTERVAL,
            "concurrent_per_key": cls.DEFAULT_CONCURRENT_PER_KEY
        }
