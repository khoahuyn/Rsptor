from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class ChunkingSettings(BaseSettings):
    """Hierarchical chunking configuration with performance optimizations"""
    model_config = ConfigDict(extra="ignore", env_file=".env", case_sensitive=False)
    
    # Basic chunk settings
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_delimiter: str = Field("\n\n", env="CHUNK_DELIMITER") 
    chunk_overlap_percent: int = Field(10, env="CHUNK_OVERLAP_PERCENT")
    hierarchical_pattern_set: int = Field(2, env="HIERARCHICAL_PATTERN_SET")
    min_chunk_tokens: int = Field(50, env="MIN_CHUNK_TOKENS")
    
    
    # Performance optimization flags
    pattern_cache_enabled: bool = Field(True)
    batch_token_counting: bool = Field(True)
    token_aware_overlap: bool = Field(True)
    
    # Concurrency and threading settings (RAGFlow-style)
    max_concurrent_chunks: int = Field(3)
    enable_thread_processing: bool = Field(True)
    enable_progress_callback: bool = Field(True)


# Global settings instance
_chunking_settings = None


def get_chunking_settings() -> ChunkingSettings:
    """Get chunking settings with caching"""
    global _chunking_settings
    
    if _chunking_settings is None:
        _chunking_settings = ChunkingSettings()
        
    return _chunking_settings
