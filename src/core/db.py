import os
import logging
from typing import Optional
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")

_engine: Optional[AsyncEngine] = None
_db_initialized = False
logger = logging.getLogger(__name__)

def get_engine() -> AsyncEngine:
    """Get or create async DB engine"""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL, 
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    return _engine

async def init_db():
    """Initialize database tables once"""
    global _db_initialized
    if _db_initialized:
        return
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Import models here to ensure they are registered before create_all
            from .models import ChatSession, ChatMessage
            await conn.run_sync(SQLModel.metadata.create_all)
        _db_initialized = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
