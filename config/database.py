from pydantic import Field
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DatabaseSettings(BaseSettings):
    
    # Database connection - Business configuration
    database_url: str = Field(..., env="DATABASE_URL")
    
    # SSL settings - Environment specific
    enable_ssl: bool = Field(True, env="DB_ENABLE_SSL")
    ssl_cert_path: str = Field("database/prod-ca-2021.crt", env="DB_SSL_CERT_PATH")
    supabase_sslrootcert: str = Field("", env="SUPABASE_SSLROOTCERT")
    
    
    # Technical defaults - Good for most cases
    echo_sql: bool = Field(False)           
    pool_pre_ping: bool = Field(True)       
    pool_timeout: int = Field(30)           
    pool_size: int = Field(10)              
    max_overflow: int = Field(20)           
    

    @property
    def vector_dimension(self) -> int:
        """Get vector dimension from embedding settings"""
        try:
            from config.embedding import get_embedding_settings
            return get_embedding_settings().embed_dimension
        except ImportError:
            return 1024  # Fallback
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        extra = "ignore"  # Ignore non-DB env vars


# Global config instance
database_settings = DatabaseSettings()

def get_database_settings() -> DatabaseSettings:
    """Get database configuration instance"""
    return database_settings
