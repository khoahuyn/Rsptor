import time
import logging
from typing import  Optional

from chat.models import SmartChatRequest, SmartChatResponse
from chat.gemini_client import GeminiChat
from config.chat import get_gemini_config
from models.tree.retrieval import RetrievalRequest
from api.ragflow_raptor import ragflow_retrieve
from prompts.chat import RAG_SYSTEM_PROMPT


class ChatService:
    def __init__(self):
        self.gemini_config = get_gemini_config()
        self.gemini_chat = GeminiChat(
            api_key=self.gemini_config.api_key,
            model_name=self.gemini_config.model_name,
            temperature=self.gemini_config.temperature,
            max_tokens=self.gemini_config.max_tokens,
            top_p=self.gemini_config.top_p,
            max_retries=self.gemini_config.max_retries
        )
        self.logger = logging.getLogger(__name__)

    async def smart_chat(self, request: SmartChatRequest) -> SmartChatResponse:
        """Simple API: query + retrieval + LLM generation in one call"""
        try:
            # Step 1: Auto retrieval with RAPTOR
            retrieval_request = RetrievalRequest(
                query=request.query,
                tenant_id=request.tenant_id,
                kb_id=request.kb_id,
                top_k=5  # Fixed reasonable value
            )
            
            retrieval_result = await ragflow_retrieve(retrieval_request)
            
            # Extract chunks from RetrievalResponse
            if hasattr(retrieval_result, 'nodes'):
                chunks = retrieval_result.nodes
            elif hasattr(retrieval_result, 'retrieved_nodes'):
                chunks = retrieval_result.retrieved_nodes
            else:
                chunks = []
            
            # Step 2: Prepare context from chunks
            if chunks:
                context_text = "\n\n".join([
                    f"[Source {i+1}]: {getattr(chunk, 'content', '')}"
                    for i, chunk in enumerate(chunks[:5])  # Top 5 chunks
                ])
                
                # Step 3: Generate answer with context using Gemini directly
                prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {request.query}"
                
                # Use Gemini generate_content directly to avoid generator issues
                response = self.gemini_chat.model.generate_content(prompt)
                answer = response.text
            else:
                # No context found - let LLM respond in user's language
                prompt = f"{RAG_SYSTEM_PROMPT}\n\nUser query: {request.query}\n\nNo relevant information found in the knowledge base."
                
                # Use Gemini generate_content directly
                response = self.gemini_chat.model.generate_content(prompt)
                answer = response.text
            
            return SmartChatResponse(answer=answer)
            
        except Exception as e:
            self.logger.error(f"Smart chat failed: {e}")
            # Let LLM handle error message in user's language
            try:
                prompt = f"{RAG_SYSTEM_PROMPT}\n\nUser query: {request.query}\n\nSystem error occurred."
                response = self.gemini_chat.model.generate_content(prompt)
                error_answer = response.text
            except:
                error_answer = "Sorry, there was a technical error. Please try again later."
            return SmartChatResponse(answer=error_answer)


# Global service instance
_chat_service: Optional[ChatService] = None


async def get_chat_service() -> ChatService:
    """Get or create chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service