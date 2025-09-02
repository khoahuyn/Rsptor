import os
from typing import Optional
from pydantic import BaseModel


class GeminiConfig(BaseModel):
    api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
            top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        )


_gemini_config: Optional[GeminiConfig] = None


def get_gemini_config() -> GeminiConfig:
    global _gemini_config
    if _gemini_config is None:
        _gemini_config = GeminiConfig.from_env()
    return _gemini_config
