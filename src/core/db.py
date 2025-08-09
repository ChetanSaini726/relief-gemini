import os
import base64
import logging
from datetime import datetime
from typing import Optional, List, Tuple

import streamlit as st
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Encryption Key Handling
# ------------------------------------------------------------------------------
def get_encryption_key() -> bytes:
    """Get encryption key from environment with validation"""
    try:
        key_b64 = os.environ.get("CHAT_DB_KEY")
        if not key_b64:
            raise ValueError("CHAT_DB_KEY environment variable is not set")
        
        key = base64.urlsafe_b64decode(key_b64)
        if len(key) not in [16, 24, 32]:
            raise ValueError("Invalid AES key length")
        
        return key
    except Exception as e:
        logger.error(f"Failed to load encryption key: {e}")
        raise

AES_KEY = get_encryption_key()

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class ChatSession(SQLModel, table=True):
    """Chat session model"""
    id: str = Field(primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(SQLModel, table=True):
    """Chat message model with session support"""
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str
    content: bytes  # encrypted blob

# ------------------------------------------------------------------------------
# Database Setup
# ------------------------------------------------------------------------------
try:
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment is not set")
except Exception as e:
    logger.error(f"Failed to load database url: {e}")
    raise

_engine: Optional[AsyncEngine] = None

def get_engine() -> AsyncEngine:
    """Get or create database engine"""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL, 
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    return _engine

_db_initialized = False

async def init_db():
    """Initialize database once"""
    global _db_initialized
    if _db_initialized:
        return
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        _db_initialized = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# ------------------------------------------------------------------------------
# Encryption Helpers
# ------------------------------------------------------------------------------
def encrypt(plaintext: str) -> bytes:
    """Encrypt plaintext with AES-GCM"""
    try:
        if not plaintext:
            return b""
        aesgcm = AESGCM(AES_KEY)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return nonce + ciphertext
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise

def decrypt(cipherbytes: bytes) -> str:
    """Decrypt ciphertext with AES-GCM"""
    try:
        if not cipherbytes:
            return ""
        if len(cipherbytes) < 12:
            raise ValueError("Invalid ciphertext length")
        aesgcm = AESGCM(AES_KEY)
        nonce, ciphertext = cipherbytes[:12], cipherbytes[12:]
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return "[Decryption error - message corrupted]"

# ------------------------------------------------------------------------------
# Database CRUD Operations
# ------------------------------------------------------------------------------
async def create_new_session(session_id: str, session_name: str):
    """Create a new chat session"""
    try:
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as db_session:
            existing = await db_session.get(ChatSession, session_id)
            if existing:
                logger.warning(f"Session {session_id} already exists")
                return
            db_session.add(ChatSession(id=session_id, name=session_name))
            await db_session.commit()
        logger.info(f"Created new session: {session_name} ({session_id})")
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise

async def get_all_sessions() -> List[ChatSession]:
    """Get all chat sessions ordered by creation date"""
    try:
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(ChatSession).order_by(ChatSession.created_at.desc())
            )
            return list(result.scalars().all())
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
        return []

async def save_message(session_id: str, role: str, content: str):
    """Save a message to the database"""
    try:
        if not session_id or not role or not content:
            raise ValueError("session_id, role, and content are required")
        if role not in ["user", "assistant"]:
            raise ValueError("role must be 'user' or 'assistant'")
        
        await init_db()
        engine = get_engine()
        encrypted_content = encrypt(content)
        message = ChatMessage(session_id=session_id, role=role, content=encrypted_content)
        
        async with AsyncSession(engine) as session:
            if not await session.get(ChatSession, session_id):
                raise ValueError(f"Session {session_id} does not exist")
            session.add(message)
            await session.commit()
        logger.info(f"Saved {role} message to session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        if hasattr(st, "error"):
            st.error(f"❌ Failed to save message")
        raise

async def load_history(session_id: str) -> List[Tuple[str, str]]:
    """Load chat history for a specific session"""
    try:
        if not session_id:
            return []
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            if not await session.get(ChatSession, session_id):
                logger.warning(f"Session {session_id} does not exist")
                return []
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id)
            )
            messages = result.scalars().all()
        return [(msg.role, decrypt(msg.content)) for msg in messages]
    except Exception as e:
        logger.error(f"Failed to load history for session {session_id}: {e}")
        if hasattr(st, "error"):
            st.error(f"❌ Failed to load chat history")
        return []

async def delete_session(session_id: str):
    """Delete a chat session and all its messages"""
    try:
        if not session_id:
            raise ValueError("session_id is required")
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            existing_session = await session.get(ChatSession, session_id)
            if existing_session:
                await session.delete(existing_session)
                await session.commit()
                logger.info(f"Deleted session {session_id}")
            else:
                logger.warning(f"Session {session_id} not found for deletion")
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise

async def get_session_by_id(session_id: str) -> Optional[ChatSession]:
    """Get a specific session by ID"""
    try:
        if not session_id:
            return None
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            return await session.get(ChatSession, session_id)
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        return None

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
async def cleanup_empty_sessions():
    """Remove sessions with no messages"""
    try:
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            all_sessions = await session.execute(select(ChatSession))
            sessions = all_sessions.scalars().all()
            empty_sessions = []
            for chat_session in sessions:
                messages = await session.execute(
                    select(ChatMessage).where(ChatMessage.session_id == chat_session.id)
                )
                if not messages.scalars().first():
                    empty_sessions.append(chat_session.id)
            for empty_id in empty_sessions:
                await delete_session(empty_id)
            logger.info(f"Cleaned up {len(empty_sessions)} empty sessions")
    except Exception as e:
        logger.error(f"Failed to cleanup empty sessions: {e}")

async def get_database_stats():
    """Get database statistics"""
    try:
        await init_db()
        engine = get_engine()
        async with AsyncSession(engine) as session:
            session_count = await session.execute(select(ChatSession))
            total_sessions = len(session_count.scalars().all())
            message_count = await session.execute(select(ChatMessage))
            total_messages = len(message_count.scalars().all())
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"total_sessions": 0, "total_messages": 0}

