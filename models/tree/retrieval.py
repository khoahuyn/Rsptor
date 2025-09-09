from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RetrievalRequest(BaseModel):
    """Request for RAGFlow embeddings-based retrieval"""
    query: str = Field(...)
    
    # RAGFlow retrieval parameters
    tenant_id: str = Field(...)
    kb_id: str = Field(...)
    
    # Common parameters
    top_k: int = Field(5, ge=1, le=50)
    token_budget: Optional[int] = Field(4000, ge=100)
    min_similarity_threshold: float = Field(0.0, ge=0.0, le=1.0)
    
    # RAGFlow-style hybrid search parameters
    candidate_multiplier: int = Field(20, ge=1, le=100)
    vector_similarity_weight: float = Field(0.7, ge=0.0, le=1.0)

class RetrievedNode(BaseModel):
    """A retrieved node with similarity score"""
    node_id: str = Field(...)
    similarity_score: float = Field(...)
    content: str = Field(...)
    level: int = Field(...)
    token_count: float = Field(...)
    meta: dict = Field(default_factory=dict)

class RetrievalStats(BaseModel):
    """Statistics about the retrieval process"""
    query_tokens: int = Field(...)
    total_candidates: int = Field(...)
    filtered_candidates: int = Field(...)
    search_method: str = Field(...)
    embedding_model: str = Field(...)
    # Cache performance metrics
    cache_hit_rate: str = Field("0%")
    cache_size: int = Field(0)

class RetrievalResponse(BaseModel):
    """Response containing retrieved nodes and stats"""
    retrieved_nodes: List[RetrievedNode] = Field(...)
    retrieval_stats: RetrievalStats = Field(...)
    
    # Deep Research metadata (optional)
    research_metadata: Optional[Dict[str, Any]] = Field(None)
    
    class Config:
        extra = "allow"  # Allow additional fields for flexibility





