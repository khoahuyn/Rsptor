import time
import asyncio
import requests
from typing import List, Dict
from exceptions import LLMError

def call_fpt_api(
    base_url: str, 
    api_key: str, 
    model: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = 2,
    timeout: int = 60
) -> str:
    """
    Simple FPT Cloud API client - returns content string only.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "streaming": False
    }
    
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error = None
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
                
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            # Success case
            if response.status_code == 200:
                try:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise LLMError(f"Invalid LLM response format: {e}", "LLM_INVALID_RESPONSE")
            
            # Retry on server errors
            elif response.status_code in [429, 500, 502, 503, 504]:
                last_error = f"HTTP {response.status_code}: {response.text}"
                if attempt < max_retries:
                    continue
            
            # Don't retry on client errors
            else:
                raise LLMError(f"HTTP {response.status_code}: {response.text}", f"HTTP_ERROR_{response.status_code}")
                
        except requests.RequestException as e:
            last_error = f"Request failed: {e}"
            if attempt < max_retries:
                continue
            break
    
    # Max retries exceeded
    raise LLMError(f"Request failed after {max_retries} retries. Last error: {last_error}", "MAX_RETRIES_EXCEEDED")


class FPTCloudClient:
    """Backward compatibility wrapper for existing code"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """Wrapper to maintain existing interface"""
        content = call_fpt_api(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 1024),
            temperature=kwargs.get('temperature', 0.7),
            max_retries=kwargs.get('max_retries', 2),
            timeout=kwargs.get('timeout', 60)
        )
        
        # Return object that matches expected interface
        return type('FPTResponse', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': content
                })()
            })()]
        })()


async def call_fpt_api_async(
    base_url: str, 
    api_key: str, 
    model: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = 2,
    timeout: int = 60
) -> str:
    """
    Async wrapper for FPT Cloud API - runs sync call in thread executor
    """
    loop = asyncio.get_event_loop()
    
    # Run sync function in thread executor to avoid blocking
    return await loop.run_in_executor(
        None, 
        call_fpt_api,
        base_url, api_key, model, messages, max_tokens, temperature, max_retries, timeout
    )

def create_fpt_client(base_url: str, api_key: str) -> FPTCloudClient:
    """Factory function to create FPT Cloud client"""
    return FPTCloudClient(base_url, api_key)

