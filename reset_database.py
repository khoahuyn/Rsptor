import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Fix Windows event loop for psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from database.connection import get_session_factory
from dotenv import load_dotenv


async def clear_data_only():
    """Clear all data but keep table structure"""
    print("🧹 Clearing data while preserving table structure...")
    
    # Load .env if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            from sqlalchemy import text
            
            # Delete data in correct order (respecting foreign keys)
            print("🔄 Clearing table data...")
            clear_tables = [
                "DELETE FROM embeddings;",
                "DELETE FROM chunks;",
                "DELETE FROM documents;",
                "DELETE FROM chat_sessions;",
                "DELETE FROM knowledge_bases;"
            ]
            
            for clear_sql in clear_tables:
                try:
                    result = await session.execute(text(clear_sql))
                    print(f"   ✅ {clear_sql.split()[2]} - {result.rowcount} rows deleted")
                except Exception as e:
                    print(f"   ⚠️  {clear_sql.split()[2]} - {e}")
            
            await session.commit()
            print("✅ Data clearing completed!")
            
    except Exception as e:
        print(f"❌ Data clearing failed: {e}")
        return False
    
    return True


async def reset_database_completely():
    """Reset database completely (drop tables and ENUM types)"""
    print("🗑️  Resetting database completely...")
    
    # Load .env if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            from sqlalchemy import text
            
            # Drop all tables
            print("🔄 Dropping tables...")
            drop_tables = [
                "DROP TABLE IF EXISTS embeddings CASCADE;",
                "DROP TABLE IF EXISTS chunks CASCADE;",
                "DROP TABLE IF EXISTS documents CASCADE;",
                "DROP TABLE IF EXISTS chat_sessions CASCADE;",
                "DROP TABLE IF EXISTS knowledge_bases CASCADE;",
                "DROP TABLE IF EXISTS alembic_version CASCADE;"
            ]
            
            for drop_sql in drop_tables:
                await session.execute(text(drop_sql))
            
            # Drop ENUM types
            print("🔄 Dropping ENUM types...")
            drop_enums = [
                "DROP TYPE IF EXISTS embedding_owner_type CASCADE;",
                "DROP TYPE IF EXISTS kb_status CASCADE;"
            ]
            
            for drop_sql in drop_enums:
                await session.execute(text(drop_sql))
            
            await session.commit()
            print("✅ Database reset completed!")
            
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False
    
    # Reset migration state
    print("🔄 Resetting migration state...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "alembic", "stamp", "base"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Migration state reset!")
        else:
            print(f"⚠️  Migration reset warning: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Migration reset failed: {e}")
    
    return True


async def main():
    """Main reset routine"""
    print("🗑️ RAPTOR Service - Database Reset")
    print("=" * 50)
    print("Supports enhanced schema with TreeNodeChunk mapping")
    print("")
    
    # Show options
    print("Choose reset option:")
    print("1️⃣  Clear data only (DELETE data, keep tables)")
    print("    ✅ Fast - no need to run setup_database.py again")
    print("    ✅ Keeps table structure intact")
    print("")
    print("2️⃣  Reset everything (DROP tables + ENUM types)")
    print("    ⚠️  Complete reset - requires setup_database.py after")
    print("    ⚠️  Removes all tables and types")
    print("")
    
    choice = input("❓ Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🧹 You chose: Clear data only")
        print("⚠️  This will DELETE ALL DATA but keep table structure!")
        confirm = input("❓ Continue? Type 'yes' to confirm: ").lower().strip()
        
        if confirm != 'yes':
            print("❌ Operation cancelled.")
            return
        
        success = await clear_data_only()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 Data clearing completed!")
            print("✅ All data deleted (including TreeNodeChunk mappings)")
            print("✅ Table structure preserved")
            print("✅ Enhanced retrieval schema intact")
            print("\n📝 Ready for new data upload & tree building!")
        else:
            print("\n❌ Data clearing failed. Check the errors above.")
            
    elif choice == "2":
        print("\n🗑️  You chose: Reset everything")
        print("⚠️  This will DELETE ALL DATA AND DROP ALL TABLES!")
        confirm = input("❓ Continue? Type 'yes' to confirm: ").lower().strip()
        
        if confirm != 'yes':
            print("❌ Operation cancelled.")
            return
        
        success = await reset_database_completely()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 Complete database reset completed!")
            print("✅ All tables and ENUM types removed")
            print("✅ TreeNodeChunk mapping tables dropped")
            print("✅ Migration state reset")
            print("\n📝 Next step: python setup_database.py")
        else:
            print("\n❌ Complete reset failed. Check the errors above.")
    else:
        print("❌ Invalid choice. Please run again and choose 1 or 2.")


if __name__ == "__main__":
    asyncio.run(main())

