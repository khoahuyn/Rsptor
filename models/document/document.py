from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of document content with embeddings"""
    chunk_id: str = Field(...)
    doc_id: str = Field(...)
    content: str = Field(...)
    chunk_index: int = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

