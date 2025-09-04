from typing import Optional

# Ensure environment variables are loaded FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from config.database import get_database_settings

# Global session factory cache
_session_factory_cache: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """Retrieve database URL from environment variables."""
    db_config = get_database_settings()
    return db_config.database_url


def get_session_factory(
    database_url: Optional[str] = None,
    enable_ssl: Optional[bool] = None,
    pool_pre_ping: bool = True
) -> async_sessionmaker[AsyncSession]:

    global _session_factory_cache
    
    # Return cached factory if available and using default parameters
    if (_session_factory_cache is not None and 
        database_url is None and 
        enable_ssl is None and 
        pool_pre_ping is True):
        return _session_factory_cache
    
    if not database_url:
        database_url = get_database_url()

    if enable_ssl is None:
        enable_ssl = get_database_settings().enable_ssl

    # SSL configuration for psycopg (like reference project)
    connect_args = {}
    if enable_ssl and "supabase" in database_url.lower():
        ssl_cert_path = get_database_settings().ssl_cert_path
        
        try:
            # Method 1: Use psycopg SSL parameters (like reference project)
            if ssl_cert_path:
                import os
                cert_path = ssl_cert_path if os.path.isabs(ssl_cert_path) else os.path.join("database", ssl_cert_path)
                
                if os.path.exists(cert_path):
                    connect_args = {
                        "sslmode": "verify-full", 
                        "sslrootcert": cert_path
                    }
                else:
                    # Fallback to require mode without certificate
                    connect_args = {"sslmode": "require"}
            else:
                # SSL require without specific certificate
                connect_args = {"sslmode": "require"}
                
        except Exception:
            try:
                # Basic SSL requirement
                connect_args = {"sslmode": "require"}
            except Exception:
                # Disable SSL as last resort
                connect_args = {}

    # Add psycopg-compatible connection parameters for concurrent operations
    connect_args.update({
        "application_name": "raptor_service",
        "prepare_threshold": None,  # Disable prepared statements to avoid conflicts in parallel uploads
    })

    # Create async engine with NullPool for async compatibility
    try:
        engine = create_async_engine(
            database_url,
            pool_pre_ping=pool_pre_ping,
            poolclass=NullPool,    # NullPool is asyncio-compatible
            echo=False,  # Disable SQL echo to avoid issues
            connect_args=connect_args
        )
        
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        
        # Cache the factory if using default parameters
        if (database_url == get_database_url() and 
            enable_ssl == get_database_settings().enable_ssl and 
            pool_pre_ping is True):
            _session_factory_cache = session_factory
            
        return session_factory
        
    except Exception:
        raise


def get_cached_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get the cached session factory, creating it if it doesn't exist.
    This is the preferred method for normal operations to maximize performance.
    """
    global _session_factory_cache
    
    if _session_factory_cache is None:
        _session_factory_cache = get_session_factory()
    
    return _session_factory_cache
