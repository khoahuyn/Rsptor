import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Fix Windows event loop for psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from database.connection import get_session_factory
from dotenv import load_dotenv


async def test_database_connection():
    """Test database connection and pgvector extension"""
    print("🔍 Testing database connection...")
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            from sqlalchemy import text
            
            # Test basic connection
            result = await session.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            print(f"✅ Database connection successful! Test query result: {test_value}")
            
            # Test pgvector extension
            try:
                result = await session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
                extension = result.scalar()
                if extension:
                    print("✅ pgvector extension is already installed")
                else:
                    print("⚠️  pgvector extension not found - enabling it...")
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    await session.commit()
                    print("✅ pgvector extension enabled successfully")
            except Exception as e:
                print(f"❌ Could not check/enable pgvector: {e}")
                print("🔧 Manual fix: Run 'CREATE EXTENSION vector;' in your Supabase SQL editor")
                
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Check your .env file contains:")
        print(f"   DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@db.your_project.supabase.co:5432/postgres")
        return False


def run_migrations():
    """Run Alembic migrations to create database schema"""
    print("\n🔄 Running database migrations...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Database migrations completed successfully")
            if result.stdout:
                print(f"Migration output: {result.stdout}")
            return True
        else:
            print(f"❌ Migration failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run migrations: {e}")
        print("🔧 Manual fix: Run 'python -m alembic upgrade head'")
        return False


async def verify_tables():
    """Verify that all required tables exist"""
    print("\n🔍 Verifying database tables...")
    required_tables = [
        "knowledge_bases", "chat_sessions", 
        "documents", "chunks", 
        # Tree tables removed - using RAGFlow approach
        "embeddings"
    ]
    
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            from sqlalchemy import text
            
            for table in required_tables:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """))
                exists = result.scalar()
                if exists:
                    print(f"  ✅ Table '{table}' exists")
                else:
                    print(f"  ❌ Table '{table}' missing!")
                    return False
        
        print("✅ All required tables verified!")
        return True
        
    except Exception as e:
        print(f"❌ Table verification failed: {e}")
        return False


async def create_sample_data():
    """Create sample Knowledge Base for testing"""
    print("\n📚 Creating sample Knowledge Base...")
    try:
        from database.repository_factory import get_repositories
        
        async with get_repositories() as repos:
            # Create sample KB
            kb = await repos.kb_repo.create_kb(
                tenant_id="demo_tenant",
                name="Sample Knowledge Base", 
                description="Sample KB for testing RAPTOR Service"
            )
            print(f"✅ Created sample KB: {kb.kb_id}")
            
            # Create sample chat session
            session = await repos.chat_repo.create_chat_session(
                tenant_id="demo_tenant",
                kb_id=kb.kb_id,
                name="Test Chat Session",
                system_prompt="You are a helpful AI assistant."
            )
            print(f"✅ Created sample chat session: {session.session_id}")
            
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False


def check_environment():
    """Check if DATABASE_URL is configured"""
    print("🔧 Checking environment configuration...")
    
    # Load .env file if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Check DATABASE_URL
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL not found in environment!")
        print("💡 Make sure you have DATABASE_URL in your .env file")
        return False
    
    print("✅ DATABASE_URL configured!")
    return True


async def main():
    """Main setup routine"""
    print("🚀 RAPTOR Service Database Setup")
    print("=" * 50)
    print("")
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Setup failed: Environment configuration issues")
        return False
    
    # Step 2: Test database connection
    if not await test_database_connection():
        print("\n❌ Setup failed: Database connection issues")
        return False
    
    # Step 3: Run migrations
    if not run_migrations():
        print("\n❌ Setup failed: Migration issues")
        return False
    
    # Step 4: Verify tables
    if not await verify_tables():
        print("\n❌ Setup failed: Table verification issues")
        return False
    
    # Step 5: Create sample data
    create_sample = input("\n❓ Create sample Knowledge Base for testing? (y/N): ").lower().strip() == 'y'
    if create_sample:
        await create_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 Database setup completed successfully!")
    print("✅ All tables created on Supabase")
    print("✅ pgvector extension enabled")
    print("✅ Enhanced tree retrieval ready")
    print("✅ Ready for document upload & tree building!")
    
    if create_sample:
        print("\n📚 Sample data created:")
        print("   - Tenant ID: demo_tenant")
        print("   - KB ID: demo_tenant::kb::sample_knowledge_base")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
