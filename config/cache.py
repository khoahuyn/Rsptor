from pydantic import Field
from pydantic_settings import BaseSettings


class CacheSettings(BaseSettings):
    """RAGFlow-style caching configuration"""
    
    # LLM Response Cache (RAGFlow approach)
    llm_cache_enabled: bool = Field(True, env="CACHE_LLM_ENABLED")
    llm_cache_max_size: int = Field(3000, env="CACHE_LLM_MAX_SIZE")
    
    # Embedding Cache (RAGFlow approach)
    embed_cache_enabled: bool = Field(True, env="CACHE_EMBED_ENABLED")
    embed_cache_max_size: int = Field(10000, env="CACHE_EMBED_MAX_SIZE")
    
    # Cache behavior with TTL support
    cache_hit_log_enabled: bool = Field(False)
    
    # TTL settings (Time To Live)
    llm_cache_ttl_seconds: int = Field(3600)  # 1 hour
    embed_cache_ttl_seconds: int = Field(86400)  # 24 hours
    
    # Cache cleanup  
    cache_auto_cleanup: bool = Field(True)
    
    # Retrieval cache settings (moved from retrieval.py)
    retrieval_cache_ttl_seconds: int = Field(300)  # 5 minutes
    retrieval_cache_max_entries: int = Field(100)
    
    # Chunking performance cache (token counting optimization)
    token_cache_enabled: bool = Field(True)
    token_cache_max_size: int = Field(10000)

    class Config:
        env_prefix = "CACHE_"
        extra = "ignore"


# Global settings instance
cache_settings = CacheSettings()

def get_cache_settings() -> CacheSettings:
    """Get cache configuration instance"""
    return cache_settings

