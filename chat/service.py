import logging
from typing import Optional

from chat.models import SmartChatRequest, SmartChatResponse, AssistantChatRequest, AssistantChatResponse
from chat.gemini_client import GeminiChat
from config.chat import get_gemini_config
from models.tree.retrieval import RetrievalRequest
from models.database.message import MessageRole
from api.ragflow_raptor import ragflow_retrieve
from database.repository_factory import get_repositories
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
    
    async def assistant_chat(self, request: AssistantChatRequest) -> AssistantChatResponse:
        """Chat with an AI assistant, save conversation to database"""
        try:
            async with get_repositories() as repos:
                # Step 1: Get assistant and verify it exists
                assistant = await repos.assistant_repo.get_assistant_with_kb(request.assistant_id)
                if not assistant:
                    raise ValueError(f"Assistant {request.assistant_id} not found")
                
                # Step 2: Get or create chat session
                session_id = request.session_id
                if not session_id:
                    # Create new session
                    session = await repos.chat_repo.create_chat_session(
                        tenant_id=assistant.tenant_id,
                        kb_id=assistant.kb_id,
                        name=f"Chat with {assistant.name}",
                        system_prompt=assistant.system_prompt,
                        settings=assistant.model_settings
                    )
                    session_id = session.session_id
                else:
                    # Verify session exists and belongs to this assistant
                    session = await repos.chat_repo.get_by_id(session_id)
                    if not session or session.assistant_id != request.assistant_id:
                        raise ValueError(f"Session {session_id} not found or not associated with assistant")
                
                # Step 3: Save user message
                user_message = await repos.message_repo.create_message(
                    session_id=session_id,
                    role=MessageRole.user,
                    content=request.query
                )
                
                # Step 4: Build conversation context if requested
                conversation_context = ""
                if request.include_context:
                    conversation_context = await repos.message_repo.get_conversation_context(
                        session_id, request.max_context_messages
                    )
                
                # Step 5: Retrieve relevant documents using assistant's KB
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    tenant_id=assistant.tenant_id,
                    kb_id=assistant.kb_id,
                    top_k=assistant.model_settings.get("top_k", 5) if assistant.model_settings else 5
                )
                
                retrieval_result = await ragflow_retrieve(retrieval_request)
                
                # Extract chunks
                if hasattr(retrieval_result, 'nodes'):
                    chunks = retrieval_result.nodes
                elif hasattr(retrieval_result, 'retrieved_nodes'):
                    chunks = retrieval_result.retrieved_nodes
                else:
                    chunks = []
                
                # Step 6: Prepare context for LLM
                system_prompt = assistant.system_prompt or RAG_SYSTEM_PROMPT
                
                # Build context sections
                context_sections = []
                
                # Add conversation history
                if conversation_context:
                    context_sections.append(f"Conversation History:\n{conversation_context}")
                
                # Add retrieved knowledge
                if chunks:
                    retrieved_context = "\n\n".join([
                        f"[Source {i+1}]: {getattr(chunk, 'content', '')}"
                        for i, chunk in enumerate(chunks[:5])
                    ])
                    context_sections.append(f"Knowledge Base Context:\n{retrieved_context}")
                
                # Combine all context
                full_context = "\n\n---\n\n".join(context_sections) if context_sections else ""
                
                # Step 7: Generate response with assistant's model settings
                model_settings = assistant.model_settings or {}
                
                if full_context:
                    prompt = f"{system_prompt}\n\nContext:\n{full_context}\n\nUser Query: {request.query}"
                else:
                    prompt = f"{system_prompt}\n\nUser Query: {request.query}"
                
                # Use assistant's temperature and other settings
                try:
                    # Configure Gemini with assistant settings
                    self.gemini_chat.temperature = model_settings.get("temperature", self.gemini_config.temperature)
                    
                    response = self.gemini_chat.model.generate_content(prompt)
                    answer = response.text
                    
                    # Extract token usage if available
                    input_tokens = len(prompt.split()) * 1.3  # Rough estimate
                    output_tokens = len(answer.split()) * 1.3
                    
                except Exception as llm_error:
                    self.logger.error(f"LLM generation failed: {llm_error}")
                    answer = "I apologize, but I encountered an error generating a response. Please try again."
                    input_tokens = output_tokens = 0
                
                # Step 8: Save assistant message
                assistant_message = await repos.message_repo.create_message(
                    session_id=session_id,
                    role=MessageRole.assistant,
                    content=answer,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    extra_metadata={
                        "model": model_settings.get("model", "gemini-1.5-flash"),
                        "temperature": model_settings.get("temperature", 0.7),
                        "chunks_used": len(chunks),
                        "context_length": len(full_context)
                    }
                )
                
                # Step 9: Update session stats (count both user + assistant messages)
                await repos.chat_repo.increment_message_count(session_id, count=2)
                
                return AssistantChatResponse(
                    answer=answer,
                    session_id=session_id,
                    message_id=assistant_message.message_id,
                    metadata={
                        "tokens": {
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                            "total": int(input_tokens + output_tokens)
                        },
                        "retrieval": {
                            "chunks_found": len(chunks),
                            "context_included": bool(conversation_context)
                        }
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Assistant chat failed: {e}")
            raise ValueError(f"Chat failed: {str(e)}")


# Global service instance
_chat_service: Optional[ChatService] = None


async def get_chat_service() -> ChatService:
    """Get or create chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service