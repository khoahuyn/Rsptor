from pydantic import BaseModel, Field


class RaptorParams(BaseModel):
    """RAGFlow-style RAPTOR parameters - clean, simplified approach"""
    
    # Core RAGFlow parameters
    max_clusters: int | None = Field(None, ge=1, le=128)
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    random_seed: int = Field(42, ge=0)
    
    # Summary parameters
    enable_summary: bool | None = None
    summary_max_tokens: int | None = Field(None, ge=100, le=4096)
    summary_prompt: str | None = None
    
    # Tree building controls
    max_levels: int = Field(10, ge=1, le=20)
    min_nodes_to_continue: int = Field(2, ge=1)
    stop_at_single_root: bool = True
    
    # Output parameters
    return_diagnostics: bool = False
    
    # Algorithm version
    algo_version: str = "ragflow@v1"





