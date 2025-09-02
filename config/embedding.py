from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingSettings(BaseSettings):
    """Multi-model embedding configuration"""
    
    # Unified embedding configuration (auto-detect provider from model)
    embed_base_url: str = Field("https://api.voyageai.com/v1", env="EMBED_BASE_URL")
    embed_api_key: str = Field("", env="EMBED_API_KEY")  # Can be single key or comma-separated keys
    embed_model: str = Field("voyage-context-3", env="EMBED_MODEL")  
    embed_dimension: int = Field(1024, env="EMBED_VECTOR_DIM")
    
    
    @property 
    def api_keys_list(self) -> list[str]:
        """Parse comma-separated API keys into list"""
        if not self.embed_api_key:
            return []
        return [key.strip() for key in self.embed_api_key.split(",") if key.strip()]
    
    @property
    def current_config(self) -> Dict[str, Any]:
        """Auto-detect provider based on model name and API key availability"""
        # Detect provider based on model name and API key
        is_voyage_model = "voyage" in self.embed_model.lower()
        has_api_key = bool(self.embed_api_key and self.embed_api_key.strip())
        
        # Use Voyage if model is voyage AND has API key
        if is_voyage_model and has_api_key:
            return {
                "provider": "voyage",
                "base_url": self.embed_base_url,
                "api_key": self.embed_api_key,
                "api_keys": self.api_keys_list,  # Multiple keys support
                "model": self.embed_model,
                "dimension": self.embed_dimension
            }
        else:
            # Fallback to BGE-M3 (local) if no API key or non-voyage model
            fallback_url = "http://localhost:11434/api/embeddings"
            fallback_model = "bge-m3:latest"
            
            return {
                "provider": "bge-m3",
                "base_url": fallback_url,
                "api_key": "",  # BGE doesn't need API key
                "model": fallback_model,
                "dimension": self.embed_dimension
            }
    
    @property
    def active_provider(self) -> str:
        """Get currently active embedding provider"""
        return self.current_config["provider"]

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global settings instance
embedding_settings = EmbeddingSettings()

def get_embedding_settings() -> EmbeddingSettings:
    """Get embedding configuration instance"""
    return embedding_settings
