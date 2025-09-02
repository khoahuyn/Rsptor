from fastapi import APIRouter, HTTPException, Depends

from chat.models import SmartChatRequest, SmartChatResponse
from chat.service import get_chat_service, ChatService


router = APIRouter(prefix="/v1/chat", tags=["Smart Chat"])


@router.post("/smart", response_model=SmartChatResponse)
async def smart_chat(
    request: SmartChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> SmartChatResponse:

    try:
        return await chat_service.smart_chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
