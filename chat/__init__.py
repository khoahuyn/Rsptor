from typing import Dict, Type

# Model registries
ChatModel: Dict[str, Type] = {}


def register_chat_model(cls):
    """Register a chat model class"""
    if hasattr(cls, '_FACTORY_NAME'):
        if isinstance(cls._FACTORY_NAME, list):
            for factory_name in cls._FACTORY_NAME:
                ChatModel[factory_name] = cls
        else:
            ChatModel[cls._FACTORY_NAME] = cls
    return cls


# Auto-import and register chat models
try:
    from .gemini_client import GeminiChat
    register_chat_model(GeminiChat)
except ImportError:
    pass

__all__ = [
    "ChatModel",
    "register_chat_model",
]
