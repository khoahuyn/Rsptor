from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    
    # FPT Cloud (DeepSeek-V3) - Business configuration
    base_url: str = Field("https://mkp-api.fptcloud.com/v1", env="LLM_BASE_URL")
    api_key: str = Field("", env="LLM_API_KEY")
    model: str = Field("DeepSeek-V3", env="LLM_MODEL")
    
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
