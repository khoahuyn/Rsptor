from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    
    # FPT Cloud - Business configuration
    base_url: str = Field("https://mkp-api.fptcloud.com/v1", env="LLM_BASE_URL")
    api_key: str = Field("", env="LLM_API_KEY")
    
    # Regular tasks (summary, etc.)
    model: str = Field("DeepSeek-V3", env="LLM_MODEL")
    
    # Primary Chat configuration (DeepSeek-R1)
    primary_chat_model: str = Field("DeepSeek-R1", env="LLM_PRIMARY_CHAT_MODEL")
    primary_chat_temperature: float = Field(0.1, env="LLM_PRIMARY_CHAT_TEMPERATURE")
    primary_chat_max_tokens: int = Field(2048, env="LLM_PRIMARY_CHAT_MAX_TOKENS")
    
    # Summary-specific parameters (RAGFlow approach)
    summary_max_tokens: int = Field(1024)
    summary_temperature: float = Field(0.3, env="LLM_SUMMARY_TEMPERATURE")  # From RAGFlow pattern
    
    # Concurrency settings (inspired by RAGFlow chat_limiter)
    llm_concurrency: int = Field(20, env="LLM_CONCURRENCY")  # Increased for RAPTOR performance
    
    @property
    def summary_prompt(self) -> str:
        try:
            from prompts.llm import RAPTOR_SUMMARY_PROMPT
            return RAPTOR_SUMMARY_PROMPT
        except ImportError:
            # Fallback if prompts module not available
            return "Summarize the content below into a structured format with JSON output."

    class Config:
        env_prefix = "LLM_"
        extra = "ignore"  # Ignore non-LLM env vars


# Global settings instance
llm_settings = LLMSettings()

def get_llm_settings() -> LLMSettings:
    """Get LLM configuration instance"""
    return llm_settings
