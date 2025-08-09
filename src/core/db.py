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
DATABASE_URL = "sqlite+aiosqlite:///chat_history.db"
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
