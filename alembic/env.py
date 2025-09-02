import asyncio
import os
import sys
from logging.config import fileConfig

# Fix Windows event loop policy for psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from models.database import Base
from database.connection import get_database_url

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Get database URL from environment or config"""
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try to get from environment first
    try:
        db_url = get_database_url()
        print(f"Using database URL: {db_url.split('@')[1] if '@' in db_url else 'localhost'}")
        return db_url
    except ValueError as e:
        print(f"Environment URL error: {e}")
        # Fallback to config if environment not set
        config_url = config.get_main_option("sqlalchemy.url")
        if config_url:
            return config_url
        else:
            raise ValueError("No database URL found in environment or config")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    try:
        url = get_url()
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )

        with context.begin_transaction():
            context.run_migrations()
    except Exception as e:
        print(f"❌ Offline migration failed: {e}")
        raise


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section) or {}
    
    # Get database URL
    db_url = get_url()
    configuration["sqlalchemy.url"] = db_url
    
    # Add SSL configuration for psycopg (like main connection.py)
    if "supabase" in db_url.lower():
        # Use psycopg SSL parameters for Supabase
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.database import get_database_settings
        import os
        
        ssl_cert_path = get_database_settings().ssl_cert_path
        connect_args = {}
        
        if ssl_cert_path:
            cert_path = ssl_cert_path if os.path.isabs(ssl_cert_path) else os.path.join("database", ssl_cert_path)
            if os.path.exists(cert_path):
                connect_args = {
                    "sslmode": "verify-full",
                    "sslrootcert": cert_path
                }
            else:
                connect_args = {"sslmode": "require"}
        else:
            connect_args = {"sslmode": "require"}
        
        connectable = async_engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            connect_args=connect_args
        )
    else:
        connectable = async_engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
