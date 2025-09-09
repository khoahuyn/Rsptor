import logging
from typing import List, Dict, Optional
from .fpt_client import call_fpt_api_async
from exceptions import LLMError

logger = logging.getLogger("deepseek_client")

class DeepSeekFPTClient:
    """
    DeepSeek-R1 client using FPT Cloud API
    Optimized for chat conversations with proper error handling
    """
    
    def __init__(self, base_url: str, api_key: str, model: str = "DeepSeek-R1"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        logger.info(f"🤖 Initialized DeepSeek-R1 client: {model}")
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60
    ) -> str:
        """
        Generate chat completion using DeepSeek-R1
        Returns clean text response
        """
        
        try:
            logger.debug(f"🚀 DeepSeek-R1 request: {len(messages)} messages, temp={temperature}")
            
            response = await call_fpt_api_async(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=timeout
            )
            
            # Clean response (remove DeepSeek artifacts)
            clean_response = self._clean_deepseek_response(response)
            
            logger.debug(f"✅ DeepSeek-R1 response: {len(clean_response)} chars")
            return clean_response
            
        except LLMError as e:
            logger.error(f"❌ DeepSeek-R1 API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ DeepSeek-R1 unexpected error: {e}")
            raise LLMError(f"DeepSeek-R1 chat failed: {str(e)}", "DEEPSEEK_ERROR")
    
    def _extract_thinking_content(self, response: str) -> str:
        """
        Extract <think> content for citation analysis
        """
        import re
        
        thinking_match = re.search(r'<think>(.*?)(?:</think>|$)', response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            logger.info(f"🧠 Extracted thinking: {len(thinking)} chars")
            logger.info(f"🧠 Thinking content: {thinking[:100]}...")
            return thinking
        logger.info("⚠️ No <think> tags found in response")
        return ""
    
    async def chat_with_thinking(self, messages, temperature=0.1, max_tokens=2048, timeout=60):
        """
        Chat completion that returns both content and thinking
        """
        try:
            response = await call_fpt_api_async(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=timeout
            )
            
            # Log raw response for debugging
            logger.info(f"📜 Raw response preview: {response[:300]}...")
            
            # Extract thinking before cleaning
            thinking_content = self._extract_thinking_content(response)
            clean_response = self._clean_deepseek_response(response)
            
            return {
                "content": clean_response,
                "thinking": thinking_content
            }
            
        except LLMError as e:
            logger.error(f"❌ DeepSeek-R1 API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ DeepSeek-R1 unexpected error: {e}")
            raise LLMError(f"DeepSeek-R1 chat failed: {str(e)}", "DEEPSEEK_ERROR")
            
        except LLMError as e:
            logger.error(f"❌ DeepSeek-R1 API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ DeepSeek-R1 unexpected error: {e}")
            raise LLMError(f"DeepSeek-R1 chat failed: {str(e)}", "DEEPSEEK_ERROR")
    
    def _clean_deepseek_response(self, response: str) -> str:
        """
        Clean DeepSeek-R1 response artifacts
        """
        import re
        
        clean_response = response.strip()
        
        # Remove thinking tags (complete or incomplete)
        clean_response = re.sub(r'<think>.*?(?:</think>|$)', '', clean_response, flags=re.DOTALL)
        
        # Remove timestamps
        clean_response = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?', '', clean_response, flags=re.IGNORECASE)
        
        # Remove markdown formatting
        clean_response = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_response)  # Remove bold **text**
        clean_response = re.sub(r'\*(.*?)\*', r'\1', clean_response)      # Remove italic *text*
        
        # Remove any remaining XML-like tags
        clean_response = re.sub(r'<[^>]*>', '', clean_response)
        
        # Clean up whitespace
        clean_response = re.sub(r'\s+', ' ', clean_response)
        clean_response = clean_response.strip()
        
        logger.debug(f"🧹 Cleaned response: {len(clean_response)} chars")
        return clean_response
    
    async def simple_chat(self, prompt: str, **kwargs) -> str:
        """
        Simple prompt-based chat (for backward compatibility)
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, **kwargs)


def create_deepseek_client(base_url: str, api_key: str, model: str = "DeepSeek-R1") -> DeepSeekFPTClient:
    """Factory function to create DeepSeek client"""
    return DeepSeekFPTClient(base_url, api_key, model)
