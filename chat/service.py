import logging
import time
from typing import Optional

from chat.models import AssistantChatRequest, AssistantChatResponse
from chat.gemini_client import GeminiChat
from config.chat import get_gemini_config
from config.llm import get_llm_settings
from llm.deepseek_client import create_deepseek_client
from models.tree.retrieval import RetrievalRequest
from models.database.message import MessageRole
from api.ragflow_raptor import ragflow_retrieve
from database.repository_factory import get_repositories
from prompts.chat import RAG_SYSTEM_PROMPT
from utils.citation_formatter import format_context_passages_for_frontend


class ChatService:
    def __init__(self):
        # Primary LLM: DeepSeek-R1 via FPT Cloud
        self.llm_config = get_llm_settings()
        self.deepseek_client = create_deepseek_client(
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
            model=self.llm_config.primary_chat_model
        )
        
        # Fallback LLM: Gemini
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
        self.logger.info(f"🤖 Chat Service initialized - Primary: {self.llm_config.primary_chat_model}, Fallback: Gemini")


    
    async def assistant_chat(self, request: AssistantChatRequest) -> AssistantChatResponse:
        """Chat with an AI assistant, save conversation to database"""
        import time
        start_time = time.time()
        chat_timings = {}  # Track chat performance
        
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
                t_retrieval = time.time()
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    tenant_id=assistant.tenant_id,
                    kb_id=assistant.kb_id,
                    top_k=assistant.model_settings.get("top_k", 8) if assistant.model_settings else 8
                )
                
                retrieval_result = await ragflow_retrieve(retrieval_request)
                chat_timings["retrieval"] = (time.time() - t_retrieval) * 1000
                
                # Extract chunks
                if hasattr(retrieval_result, 'nodes'):
                    chunks = retrieval_result.nodes
                elif hasattr(retrieval_result, 'retrieved_nodes'):
                    chunks = retrieval_result.retrieved_nodes
                else:
                    chunks = []
                
                
                # Step 6: Get model settings
                model_settings = assistant.model_settings or {}
                
                # Step 7: Generate response with Gemini LLM
                # Prepare context for LLM
                system_prompt = assistant.system_prompt or RAG_SYSTEM_PROMPT
            
                # Build context sections
                context_sections = []
            
                # Add conversation history
                if conversation_context:
                    context_sections.append(f"Conversation History:\n{conversation_context}")
            
                # Add retrieved knowledge with proper source IDs
                if chunks:
                    retrieved_context_parts = []
                    for chunk in chunks[:5]:
                        # Create meaningful source ID from chunk metadata
                        doc_id = getattr(chunk, 'meta', {}).get('doc_id', 'unknown') if hasattr(chunk, 'meta') else 'unknown'
                        chunk_index = getattr(chunk, 'meta', {}).get('chunk_index', 0) if hasattr(chunk, 'meta') else 0
                        source_id = f"{doc_id}_chunk_{chunk_index}"
                        
                        retrieved_context_parts.append(f"[Source: {source_id}]\n{chunk.content}")
                    
                    retrieved_context = "\n\n".join(retrieved_context_parts)
                    context_sections.append(f"Knowledge Base Context:\n{retrieved_context}")
            
                # Combine all context
                full_context = "\n\n---\n\n".join(context_sections) if context_sections else ""
            
                if full_context:
                    prompt = f"{system_prompt}\n\nContext:\n{full_context}\n\nUser Query: {request.query}"
                else:
                    prompt = f"{system_prompt}\n\nUser Query: {request.query}"
            
                # Step 7.1: Try DeepSeek-R1 first (Primary LLM)
                answer = None
                input_tokens = output_tokens = 0
                used_model = "unknown"
                thinking_content = ""  # Store thinking for citation analysis
                
                try:
                    self.logger.info("🤖 Trying DeepSeek-R1 (Primary LLM)")
                    
                    # Prepare messages for DeepSeek
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{full_context}\n\nUser Query: {request.query}" if full_context else request.query}
                    ]
                    
                    t_llm = time.time()
                    deepseek_response = await self.deepseek_client.chat_with_thinking(
                        messages=messages,
                        temperature=model_settings.get("temperature", self.llm_config.primary_chat_temperature),
                        max_tokens=model_settings.get("max_tokens", self.llm_config.primary_chat_max_tokens)
                    )
                    answer = deepseek_response["content"]
                    thinking_content = deepseek_response["thinking"]
                    chat_timings["llm_response"] = (time.time() - t_llm) * 1000
                    
                    # Estimate tokens for DeepSeek
                    input_tokens = sum(len(msg["content"].split()) for msg in messages) * 1.3
                    output_tokens = len(answer.split()) * 1.3
                    used_model = self.llm_config.primary_chat_model
                    
                    self.logger.info(f"✅ DeepSeek-R1 success: {len(answer)} chars")
                    if thinking_content:
                        self.logger.info(f"🧠 Thinking extracted: {len(thinking_content)} chars")
                        self.logger.info(f"🧠 Thinking preview: {thinking_content[:200]}...")
                    else:
                        self.logger.info("⚠️ No thinking content extracted from DeepSeek response")
                    
                except Exception as deepseek_error:
                    self.logger.warning(f"⚠️ DeepSeek-R1 failed: {deepseek_error}")
                    
                    # Step 7.2: Fallback to Gemini
                    try:
                        self.logger.info("🔄 Falling back to Gemini")
                        
                        # Configure Gemini with assistant settings
                        self.gemini_chat.temperature = model_settings.get("temperature", self.gemini_config.temperature)
                        
                        t_llm = time.time()
                        response = self.gemini_chat.model.generate_content(prompt)
                        answer = response.text
                        chat_timings["llm_response"] = (time.time() - t_llm) * 1000
                        
                        # Extract token usage for Gemini
                        input_tokens = len(prompt.split()) * 1.3
                        output_tokens = len(answer.split()) * 1.3
                        used_model = self.gemini_config.model_name + " (fallback)"
                        thinking_content = ""  # No thinking from Gemini
                        
                        self.logger.info(f"✅ Gemini fallback success: {len(answer)} chars")
                        
                    except Exception as gemini_error:
                        self.logger.error(f"❌ Both LLMs failed - DeepSeek: {deepseek_error}, Gemini: {gemini_error}")
                        answer = "I apologize, but I encountered an error generating a response. Please try again."
                        input_tokens = output_tokens = 0
                        used_model = "error"
                        thinking_content = ""  # No thinking available on error
                
                # Ensure we always have some response
                if not answer or not answer.strip():
                    self.logger.warning("⚠️ Empty response detected, providing fallback")
                    answer = "I don't have enough information in the knowledge base to answer this question."
                    used_model += " (fallback)"
                
                # Use model's thinking to decide citations (much smarter!)
                self.logger.info("🎯 Analyzing model thinking for citation decision...")
                should_show_citations = _should_show_citations_from_thinking(thinking_content, answer)
                self.logger.info(f"📊 Citation decision: {should_show_citations}")
                if should_show_citations:
                    self.logger.info(f"✅ Will show citations - model referenced chunks in thinking")
                else:
                    self.logger.info(f"🚫 No citations - model didn't reference specific chunks")
                self.logger.info(f"📄 Answer preview: {answer[:200]}...")
                
                # Step 8: Save assistant message
                assistant_message = await repos.message_repo.create_message(
                    session_id=session_id,
                    role=MessageRole.assistant,
                    content=answer,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    extra_metadata={
                        "model": used_model,
                        "temperature": model_settings.get("temperature", 0.7),
                        "chunks_used": len(chunks),
                        "context_length": len(full_context),
                        "llm_strategy": "deepseek_primary_gemini_fallback"
                    }
                )
                
                # Step 9: Update session stats (count both user + assistant messages)
                await repos.chat_repo.increment_message_count(session_id, count=2)
                
                # Calculate total chat time
                total_time = (time.time() - start_time) * 1000
                chat_timings["total"] = total_time
                
                # Log performance breakdown
                self.logger.info(f"🚀 Chat Performance Breakdown:")
                self.logger.info(f"   • Retrieval: {chat_timings.get('retrieval', 0):.1f}ms")
                self.logger.info(f"   • LLM Response ({used_model}): {chat_timings.get('llm_response', 0):.1f}ms") 
                self.logger.info(f"   • Total: {total_time:.1f}ms")
                
                # Generate context passages for response (single call)
                final_context_passages = self._debug_format_context_passages(
                    chunks, should_show_citations, request.query, thinking_content, answer
                )
                
                self.logger.info(f"🎯 Final response will include {len(final_context_passages)} context passages")
                
                return AssistantChatResponse(
                    answer=answer,
                    session_id=session_id,
                    message_id=assistant_message.message_id,
                    context_passages=final_context_passages,  
                    metadata={
                        "tokens": {
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                            "total": int(input_tokens + output_tokens)
                        },
                        "retrieval": {
                            "chunks_found": len(chunks),
                            "context_included": bool(conversation_context),
                            "citations_included": should_show_citations
                        },
                        "llm": {
                            "model_used": used_model,
                            "strategy": "deepseek_primary_gemini_fallback"
                        },
                        "performance": chat_timings  # 🚀 ADD: Performance metrics
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Assistant chat failed: {e}")
            raise ValueError(f"Chat failed: {str(e)}")
    
    def _debug_format_context_passages(self, chunks, should_show_citations, user_query, thinking_content, answer):
        """Debug helper to trace context passage generation"""
        self.logger.info(f"🔍 Context passage debug:")
        self.logger.info(f"   • Chunks available: {len(chunks) if chunks else 0}")
        self.logger.info(f"   • Should show citations: {should_show_citations}")
        
        if should_show_citations and chunks:
            from utils.citation_formatter import format_context_passages_for_frontend
            result = format_context_passages_for_frontend(
                retrieved_nodes=chunks,
                user_query=user_query,
                thinking_content=thinking_content,
                model_answer=answer
            )
            self.logger.info(f"   • Generated passages: {len(result)}")
            if result:
                for i, passage in enumerate(result):
                    self.logger.info(f"     {i+1}. {passage['source_id']}: {passage['relevant_excerpt'][:80]}...")
            else:
                self.logger.warning("   ⚠️ format_context_passages_for_frontend returned empty list!")
            return result
        else:
            self.logger.info(f"   • Skipping: should_show={should_show_citations}, chunks={len(chunks) if chunks else 0}")
            return []

def _should_show_citations_from_thinking(thinking_content: str, answer: str) -> bool:
    """
    Decide citations based on model's thinking - much simpler and more accurate!
    If model referenced chunks in thinking → show citations
    If model said no info → no citations
    """
    import logging
    logger = logging.getLogger("chat.service")
    
    if not thinking_content:
        logger.info("🤷 No thinking content available - defaulting to no citations")
        return False
    
    thinking_lower = thinking_content.lower()
    answer_lower = answer.lower()
    
    # Check for "no information" responses first
    no_info_phrases = [
        "don't have enough information", "does not contain", 
        "no information", "not available", "cannot find",
        "no relevant information", "not mentioned"
    ]
    
    if any(phrase in answer_lower for phrase in no_info_phrases):
        logger.info("🚫 Answer indicates no information available")
        return False
    
    # Dynamic detection of chunk references - no hardcoded patterns
    import re
    
    # Primary indicators: explicit chunk mentions
    explicit_chunks = re.findall(r'chunk[_\s]*(\d+)', thinking_lower)
    
    # Secondary indicators: reference language patterns (dynamic detection)
    reference_indicators = []
    
    # Look for any mention of "chunk" with context
    if 'chunk' in thinking_lower:
        reference_indicators.append('chunk_reference')
    
    # Look for analysis language
    analysis_terms = ['mentions', 'states', 'shows', 'according', 'based on', 'looking at', 'checking']
    for term in analysis_terms:
        if term in thinking_lower:
            reference_indicators.append(f'analysis_{term}')
    
    # Look for source/context references  
    if any(word in thinking_lower for word in ['context', 'sources', 'knowledge', 'provided']):
        reference_indicators.append('source_reference')
    
    chunks_referenced = len(explicit_chunks) > 0 or len(reference_indicators) >= 2
    found_patterns = reference_indicators[:3]  # Top patterns
    
    # Use explicit chunks as matches  
    chunk_matches = explicit_chunks
    
    if chunks_referenced:
        logger.info(f"✅ Model referenced chunks in thinking")
        logger.info(f"🔍 Found patterns: {found_patterns[:3]}")
        if chunk_matches:
            logger.info(f"📋 Specific chunks: {chunk_matches[:3]}")
        return True
    else:
        logger.info("❌ Model didn't reference any chunks in thinking")
        logger.info(f"💭 Thinking preview: {thinking_content[:150]}...")
        return False


# Global service instance
_chat_service: Optional[ChatService] = None


async def get_chat_service() -> ChatService:
    """Get or create chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service