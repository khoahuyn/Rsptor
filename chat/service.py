import logging
from typing import Optional, List, Dict, Any

from chat.models import AssistantChatRequest, AssistantChatResponse
from chat.gemini_client import GeminiChat
from config.chat import get_gemini_config
from models.tree.retrieval import RetrievalRequest
from models.database.message import MessageRole
from api.ragflow_raptor import ragflow_retrieve
from database.repository_factory import get_repositories
from prompts.chat import RAG_SYSTEM_PROMPT, EXCERPT_EXTRACTION_PROMPT


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


    async def _llm_extract_relevant_excerpts(self, content: str, answer: str) -> str:
        """Use LLM to intelligently extract relevant excerpts from content based on answer"""
        if not content or not answer:
            return ""
        
        # Use extraction prompt from prompts/chat.py
        extraction_prompt = EXCERPT_EXTRACTION_PROMPT.format(answer=answer, content=content)

        try:
            # Simplified approach - use just the user message
            user_message = f"You are an expert at extracting relevant information from documents. Return only the requested excerpts, no explanations.\n\n{extraction_prompt}"
            
            # Try the direct model approach to avoid chat wrapper issues
            import google.generativeai as genai
            
            # Configure with API key if not already done
            genai.configure(api_key=self.gemini_chat.api_key)
            
            # Configure the model directly 
            model = genai.GenerativeModel(self.gemini_chat.model_name)
            
            # Generate response directly
            response = model.generate_content(
                user_message,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200
                }
            )
            
            # Handle direct Gemini response
            self.logger.debug(f"Direct Gemini response type: {type(response)}")
            
            if hasattr(response, 'text') and response.text:
                extracted_text = response.text
                self.logger.debug(f"Using response.text: {extracted_text[:100]}...")
            elif hasattr(response, 'content'):
                extracted_text = response.content
                self.logger.debug(f"Using response.content: {extracted_text[:100]}...")
            else:
                # Fallback
                extracted_text = str(response)
                self.logger.debug(f"Fallback to str: {extracted_text[:100]}...")
            
            # Clean up the response
            cleaned_excerpt = extracted_text.strip()
            
            # Remove any unwanted prefixes that might be added by LLM
            unwanted_prefixes = ["RELEVANT EXCERPTS:", "Excerpts:", "The relevant excerpts are:", "Based on the answer:"]
            for prefix in unwanted_prefixes:
                if cleaned_excerpt.startswith(prefix):
                    cleaned_excerpt = cleaned_excerpt[len(prefix):].strip()
            
            # Fallback to simple extraction if LLM fails or returns empty
            if not cleaned_excerpt or len(cleaned_excerpt) < 10:
                # Simple fallback: take first 250 characters of content
                return content[:250].strip() + "..." if len(content) > 250 else content.strip()
            
            # Limit length if needed
            if len(cleaned_excerpt) > 300:
                # Try to cut at sentence boundary
                sentences = cleaned_excerpt.split('.')
                result = ""
                for sentence in sentences:
                    if len(result + sentence + ".") <= 300:
                        result += sentence + "."
                    else:
                        break
                cleaned_excerpt = result.strip()
            
            return cleaned_excerpt
            
        except Exception as e:
            self.logger.warning(f"LLM excerpt extraction failed: {e} (response type: {type(response) if 'response' in locals() else 'unknown'}), falling back to smart extraction")
            
            # Smart fallback: find sentences containing key information
            return self._smart_fallback_extraction(content, answer)

    def _smart_fallback_extraction(self, content: str, answer: str) -> str:
        """Smart fallback extraction when LLM fails - find relevant sentences"""
        try:
            # Extract potential keywords from answer
            answer_keywords = []
            if answer:
                # Simple keyword extraction from answer
                import re
                words = re.findall(r'\b\w+\b', answer.lower())
                # Filter meaningful words (>2 chars, not common stop words)
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'was', 'is', 'are', 'were'}
                answer_keywords = [w for w in words if len(w) > 2 and w not in stop_words]
            
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter very short sentences
            
            if not sentences:
                return content[:250].strip() + "..." if len(content) > 250 else content.strip()
            
            # Score sentences based on keyword matches
            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = 0
                
                # Score based on answer keywords
                for keyword in answer_keywords:
                    if keyword in sentence_lower:
                        score += 2  # Higher weight for answer keywords
                
                # Score based on common important keywords
                important_keywords = ['championship', 'won', 'winner', 'defeated', 'final', 'champion', '2017']
                for keyword in important_keywords:
                    if keyword in sentence_lower:
                        score += 1
                
                scored_sentences.append((score, sentence))
            
            # Sort by score and get best sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            # Return best sentence(s)
            if scored_sentences and scored_sentences[0][0] > 0:
                # Take top sentence(s) that have some relevance
                best_sentences = []
                max_length = 300
                current_length = 0
                
                for score, sentence in scored_sentences:
                    if score <= 0:
                        break
                    if current_length + len(sentence) <= max_length:
                        best_sentences.append(sentence)
                        current_length += len(sentence)
                    else:
                        break
                
                if best_sentences:
                    return ". ".join(best_sentences).strip() + "."
            
            # Fallback: return first sentence if it's informative
            if sentences and len(sentences[0]) > 30:
                return sentences[0].strip() + "."
            
            # Final fallback: first 250 chars
            return content[:250].strip() + "..." if len(content) > 250 else content.strip()
            
        except Exception as fallback_error:
            self.logger.warning(f"Smart fallback extraction failed: {fallback_error}, using simple fallback")
            return content[:250].strip() + "..." if len(content) > 250 else content.strip()

    async def _format_context_passages(self, chunks, answer: str = "") -> List[Dict[str, Any]]:
        """Format chunks as context passages for frontend citations"""
        if not chunks:
            return []
        
        # Only take the top chunk (highest similarity)
        top_chunk = chunks[0] if chunks else None
        if not top_chunk:
            return []
        
        # Extract relevant excerpt(s) using LLM - much smarter than hardcoded logic
        content = top_chunk.content
        relevant_excerpt = await self._llm_extract_relevant_excerpts(content, answer)
        
        passage = {
            "content": content,  # Keep full content for potential future use
            "relevant_excerpt": relevant_excerpt,  # ✅ ADD: Smart excerpt
            "similarity_score": top_chunk.similarity_score
        }
        
        # Add metadata if available
        if hasattr(top_chunk, 'meta') and top_chunk.meta:
            passage.update({
                "doc_id": top_chunk.meta.get('doc_id'),
                "chunk_index": top_chunk.meta.get('chunk_index'),
                "owner_type": top_chunk.meta.get('owner_type')
            })
        
        return [passage]  # ✅ Return only 1 passage

    
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
                    top_k=assistant.model_settings.get("top_k", 5) if assistant.model_settings else 5
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
                        f"[Source {i+1}]: {chunk.content}"
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
                    
                    t_llm = time.time()
                    response = self.gemini_chat.model.generate_content(prompt)
                    answer = response.text
                    chat_timings["llm_response"] = (time.time() - t_llm) * 1000
                    
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
                        "context_length": len(full_context),
                        # ✅ ADD: Store context passages for frontend retrieval
                        "context_passages": await self._format_context_passages(chunks, answer)
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
                self.logger.info(f"   • LLM Response: {chat_timings.get('llm_response', 0):.1f}ms") 
                self.logger.info(f"   • Total: {total_time:.1f}ms")
                
                return AssistantChatResponse(
                    answer=answer,
                    session_id=session_id,
                    message_id=assistant_message.message_id,
                    context_passages=await self._format_context_passages(chunks, answer),  
                    metadata={
                        "tokens": {
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                            "total": int(input_tokens + output_tokens)
                        },
                        "retrieval": {
                            "chunks_found": len(chunks),
                            "context_included": bool(conversation_context)
                        },
                        "performance": chat_timings  # 🚀 ADD: Performance metrics
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