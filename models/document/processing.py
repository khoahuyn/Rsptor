from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field



class DocumentProcessSummary(BaseModel):
    """Summary of document processing results"""
    doc_id: str = Field(...)
    filename: str = Field(...)
    total_chunks: int = Field(...)
    total_embeddings: int = Field(...)
    processing_time: float = Field(...)
    status: str = Field(...)
    
    # RAPTOR-specific chunk counts
    original_chunks: Optional[int] = Field(default=None)
    summary_chunks: Optional[int] = Field(default=None)
    
    # Optional RAPTOR fields
    raptor_enabled: bool = Field(default=False)
    raptor_summary_count: Optional[int] = Field(default=None)
    raptor_tree_levels: Optional[int] = Field(default=None)
    
    # Metadata
    tenant_id: str = Field(...)
    kb_id: str = Field(...)
    created_at: Optional[str] = Field(default=None)
    

