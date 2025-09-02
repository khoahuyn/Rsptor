from typing import Dict, List
from pydantic import Field
from pydantic_settings import BaseSettings


class FileSettings(BaseSettings):
    
    # File upload limits 
    max_file_size_mb: int = Field(50, env="FILE_MAX_SIZE_MB")
    supported_extensions: List[str] = Field([".md", ".markdown"])  
    
    # Content limits 
    min_content_length: int = Field(10)
    max_content_length: int = Field(1000000)

    @property
    def max_file_size_bytes(self) -> int:
        """Calculate max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024

    class Config:
        env_prefix = "FILE_"
        env_file = ".env"
        extra = "ignore"  # Ignore non-FILE env vars


class FileConstants:
    """File-related static constants and utilities"""
    
    # Environment variables (for validation)
    REQUIRED_ENV_VARS = []  # EMBED_API_KEY is conditionally checked based on model
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get supported extensions from dynamic settings"""
        return get_file_settings().supported_extensions
    
    @classmethod
    def is_supported_file(cls, filename: str) -> bool:
        """Check if file extension is supported"""
        if not filename:
            return False
        # Use dynamic settings for extensions
        settings = get_file_settings()
        return any(filename.lower().endswith(ext) for ext in settings.supported_extensions)
    
    @classmethod
    def get_content_limits(cls) -> Dict[str, int]:
        """Get content validation limits from dynamic settings"""
        settings = get_file_settings()
        return {
            "min_length": settings.min_content_length,
            "max_length": settings.max_content_length,
            "max_file_size": settings.max_file_size_bytes
        }


# Settings instance and getter
_file_settings = None

def get_file_settings() -> FileSettings:
    """Get file settings singleton"""
    global _file_settings
    if _file_settings is None:
        _file_settings = FileSettings()
    return _file_settings
