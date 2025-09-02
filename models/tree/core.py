from pydantic import BaseModel, Field
from typing import List



class SummaryOutput(BaseModel):
    """RAGFlow-style simple summarization - just content string."""
    content: str = Field(min_length=10, max_length=2000)





