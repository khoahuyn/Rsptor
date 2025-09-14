from pydantic import BaseModel, Field
from typing import Optional


class RetrievalConfig(BaseModel):
    
    # Similarity calculation weight (text_weight = 1.0 - vector_weight)
    vector_similarity_weight: float = Field(0.95)
    
    # Token estimation
    tokens_per_word: float = Field(1.3)

    early_exit_threshold: float = Field(0.4)


# Global retrieval configuration
_retrieval_config: Optional[RetrievalConfig] = None


def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration instance"""
    global _retrieval_config
    if _retrieval_config is None:
        _retrieval_config = RetrievalConfig()
    return _retrieval_config


def set_retrieval_config(config: RetrievalConfig):
    """Set custom retrieval configuration"""
    global _retrieval_config
    _retrieval_config = config
