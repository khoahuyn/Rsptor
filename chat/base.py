import os
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union
from enum import Enum


class ChatErrorCode(str, Enum):
    ERROR_RATE_LIMIT = "RATE_LIMIT_EXCEEDED"
    ERROR_AUTHENTICATION = "AUTH_ERROR"
    ERROR_INVALID_REQUEST = "INVALID_REQUEST"
    ERROR_SERVER = "SERVER_ERROR"
    ERROR_TIMEOUT = "TIMEOUT"
    ERROR_CONNECTION = "CONNECTION_ERROR"
    ERROR_MODEL = "MODEL_ERROR"
    ERROR_QUOTA = "QUOTA_EXCEEDED"
    ERROR_MAX_RETRIES = "MAX_RETRIES_EXCEEDED"
    ERROR_GENERIC = "GENERIC_ERROR"


class Message:
    """Chat message representation"""
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content
        self.name = name
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


class ChatResponse:
    """Chat completion response"""
    def __init__(self, 
                 content: str, 
                 model: str, 
                 usage: Optional[Dict[str, int]] = None,
                 finish_reason: str = "stop"):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.created = int(time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{random.randint(100000, 999999)}",
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.content},
                "finish_reason": self.finish_reason
            }],
            "usage": self.usage
        }


class BaseChatModel(ABC):
    """Base class for chat models, inspired by RAGFlow's Base class"""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        
        # Retry configuration
        self.max_retries = kwargs.get("max_retries", int(os.environ.get("LLM_MAX_RETRIES", 3)))
        self.base_delay = kwargs.get("retry_interval", float(os.environ.get("LLM_BASE_DELAY", 1.0)))
        
        # Model parameters
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 0.9)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_delay(self) -> float:
        """Calculate retry delay with jitter"""
        return self.base_delay * random.uniform(1.0, 2.0)
    
    def _classify_error(self, error: Exception) -> ChatErrorCode:
        """Classify error based on error message"""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["rate limit", "429", "too many requests"]):
            return ChatErrorCode.ERROR_RATE_LIMIT
        elif any(keyword in error_str for keyword in ["auth", "key", "401", "forbidden"]):
            return ChatErrorCode.ERROR_AUTHENTICATION
        elif any(keyword in error_str for keyword in ["invalid", "400", "bad request"]):
            return ChatErrorCode.ERROR_INVALID_REQUEST
        elif any(keyword in error_str for keyword in ["quota", "billing", "credit"]):
            return ChatErrorCode.ERROR_QUOTA
        elif any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return ChatErrorCode.ERROR_TIMEOUT
        elif any(keyword in error_str for keyword in ["server", "503", "502", "500"]):
            return ChatErrorCode.ERROR_SERVER
        else:
            return ChatErrorCode.ERROR_GENERIC
    
    @abstractmethod
    def chat(self, 
             messages: List[Union[Message, Dict[str, str]]], 
             stream: bool = False,
             **kwargs) -> Union[ChatResponse, Generator[str, None, None]]:
        """
        Generate chat completion
        
        Args:
            messages: List of messages
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse or Generator for streaming
        """
        pass
    
    def _format_messages(self, messages: List[Union[Message, Dict[str, str]]]) -> List[Dict[str, str]]:
        """Convert messages to standard format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, Message):
                formatted.append(msg.to_dict())
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return formatted
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry and exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_code = self._classify_error(e)
                
                # Don't retry for certain errors
                if error_code in [ChatErrorCode.ERROR_AUTHENTICATION, 
                                ChatErrorCode.ERROR_INVALID_REQUEST]:
                    raise e
                
                if attempt < self.max_retries:
                    delay = self._get_delay() * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
