from typing import List, Dict, Any, Union, Generator, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from chat.base import BaseChatModel, Message, ChatResponse, ChatErrorCode


class GeminiChat(BaseChatModel):
    """
    Gemini AI chat model implementation
    Follows RAGFlow's pattern with _FACTORY_NAME for auto-registration
    """
    
    # Factory name for auto-registration (RAGFlow pattern)
    _FACTORY_NAME = ["gemini", "google-gemini", "google"]
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
        }
        
        # Safety settings - disable để test dễ hơn
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.logger.info(f"Initialized Gemini model: {model_name}")
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI-style messages to Gemini format
        Gemini uses 'user' and 'model' roles instead of 'user' and 'assistant'
        """
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert roles
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if gemini_messages and gemini_messages[0]["role"] == "user":
                    gemini_messages[0]["parts"][0] = f"System: {content}\n\nUser: {gemini_messages[0]['parts'][0]}"
                else:
                    gemini_messages.insert(0, {
                        "role": "user",
                        "parts": [f"System: {content}\n\nPlease follow these instructions."]
                    })
            elif role == "user":
                gemini_messages.append({
                    "role": "user", 
                    "parts": [content]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",  # Gemini uses 'model' instead of 'assistant'
                    "parts": [content]
                })
        
        return gemini_messages
    
    def _extract_usage_stats(self, response) -> Dict[str, int]:
        """Extract token usage from Gemini response"""
        try:
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                return {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
        except Exception as e:
            self.logger.warning(f"Could not extract usage stats: {e}")
        
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def chat(self, 
             messages: List[Union[Message, Dict[str, str]]], 
             stream: bool = False,
             **kwargs) -> Union[ChatResponse, Generator[str, None, None]]:
        """
        Generate chat completion using Gemini
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse object or Generator for streaming
        """
        # Format messages
        formatted_messages = self._format_messages(messages)
        gemini_messages = self._convert_messages_to_gemini_format(formatted_messages)
        
        # Override generation config if provided
        generation_config = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs["top_p"]
        
        def _generate():
            try:
                if stream:
                    # Streaming response
                    response = self.model.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        stream=True
                    )
                    
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                else:
                    # Non-streaming response
                    response = self.model.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        stream=False
                    )
                    
                    content = response.text if response.text else ""
                    usage = self._extract_usage_stats(response)
                    
                    return ChatResponse(
                        content=content,
                        model=self.model_name,
                        usage=usage,
                        finish_reason="stop"
                    )
                    
            except Exception as e:
                self.logger.error(f"Gemini API error: {e}")
                error_code = self._classify_error(e)
                
                if error_code == ChatErrorCode.ERROR_RATE_LIMIT:
                    raise Exception(f"Rate limit exceeded: {e}")
                elif error_code == ChatErrorCode.ERROR_AUTHENTICATION:
                    raise Exception(f"Authentication failed: {e}")
                elif error_code == ChatErrorCode.ERROR_QUOTA:
                    raise Exception(f"Quota exceeded: {e}")
                else:
                    raise Exception(f"Gemini error: {e}")
        
        # Use retry mechanism for non-streaming
        if stream:
            return _generate()
        else:
            return self._retry_with_backoff(_generate)
    
    def simple_chat(self, prompt: str, **kwargs) -> str:
        """
        Simple chat interface for single prompt
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, stream=False, **kwargs)
        return response.content
    
    def chat_with_context(self, 
                         query: str, 
                         context: str, 
                         system_prompt: Optional[str] = None,
                         **kwargs) -> str:
        """
        Chat with retrieved context (perfect for RAG)
        
        Args:
            query: User question
            context: Retrieved context from RAPTOR
            system_prompt: Optional system instruction
            **kwargs: Additional parameters
            
        Returns:
            Generated answer
        """
        # Default system prompt for RAG
        if system_prompt is None:
            from ..prompts.chat import RAG_SYSTEM_PROMPT
            system_prompt = RAG_SYSTEM_PROMPT
        
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = self.chat(messages, stream=False, **kwargs)
        return response.content
