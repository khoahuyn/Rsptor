from pydantic import BaseModel


class SmartChatRequest(BaseModel):
    query: str
    tenant_id: str
    kb_id: str


class SmartChatResponse(BaseModel):
    answer: str
