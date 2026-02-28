import asyncio
import os
from sqlalchemy import text
from backend.api.database import engine

async def migrate():
    print("🚀 Starting database migration...")
    async with engine.begin() as conn:
        try:
            # Check if column exists
            check_sql = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='chat_history' AND column_name='citations';
            """)
            result = await conn.execute(check_sql)
            exists = result.fetchone()
            
            if not exists:
                print("➕ Adding 'citations' column to 'chat_history' table...")
                await conn.execute(text("ALTER TABLE chat_history ADD COLUMN citations JSONB DEFAULT '[]'::jsonb;"))
                print("✅ Migration successful!")
            else:
                print("ℹ️ Column 'citations' already exists.")
        except Exception as e:
            print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
