from dataclasses import dataclass
from pydantic import Field
from pydantic_settings import BaseSettings


class RaptorSettings(BaseSettings):
    """RAPTOR algorithm configuration - RAGFlow approach"""
    
    # Core RAPTOR parameters (tunable)
    max_clusters: int = Field(64, env="RAPTOR_MAX_CLUSTERS")
    similarity_threshold: float = Field(0.1, env="RAPTOR_SIMILARITY_THRESHOLD")
    
    # Technical parameters (good defaults)
    max_levels: int = Field(10)
    random_seed: int = Field(42)
    
    umap_n_components: int = Field(12)  
    umap_metric: str = Field("cosine")  

    class Config:
        env_prefix = "RAPTOR_"
        extra = "ignore"


@dataclass
class RaptorPolicy:
    """Simplified policy for RAGFlow approach"""
    max_clusters: int
    similarity_threshold: float
    max_levels: int
    random_seed: int
    umap_n_components: int
    umap_metric: str


# Global settings instance
raptor_settings = RaptorSettings()

def get_raptor_settings() -> RaptorSettings:
    """Get RAPTOR configuration instance"""
    return raptor_settings

def get_raptor_policy() -> RaptorPolicy:
    """Get simplified RAPTOR policy from settings"""
    settings = get_raptor_settings()
    
    return RaptorPolicy(
        max_clusters=settings.max_clusters,
        similarity_threshold=settings.similarity_threshold,
        max_levels=settings.max_levels,
        random_seed=settings.random_seed,
        umap_n_components=settings.umap_n_components,
        umap_metric=settings.umap_metric
    )
