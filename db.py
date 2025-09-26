import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Use DATABASE_URL from environment, fallback to local connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:9515@localhost:5432/gigaspace"
)

pool = None

async def init_db_pool():
    """
    Initialize the asyncpg connection pool (only once).
    """
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(DATABASE_URL)
    return pool

async def get_db_pool():
    """
    Return the connection pool. Call init_db_pool() if not yet created.
    """
    global pool
    if pool is None:
        pool = await init_db_pool()
    return pool
