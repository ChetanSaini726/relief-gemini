import os
import base64
import logging
from typing import Optional, List, Tuple
import streamlit as st
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .models import ChatSession, ChatMessage
from .db import get_engine, init_db

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Encryption
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

def encrypt(plaintext: str) -> bytes:
    """Encrypt plaintext"""
    if not plaintext:
        return b""
    aesgcm = AESGCM(AES_KEY)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext

def decrypt(cipherbytes: bytes) -> str:
    """Decrypt ciphertext"""
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
# CRUD Operations
# ------------------------------------------------------------------------------
async def create_new_session(session_id: str, session_name: str):
    await init_db()
    engine = get_engine()
    async with AsyncSession(engine) as db_session:
        if await db_session.get(ChatSession, session_id):
            logger.warning(f"Session {session_id} already exists")
            return
        db_session.add(ChatSession(id=session_id, name=session_name))
        await db_session.commit()

async def get_all_sessions() -> List[ChatSession]:
    await init_db()
    engine = get_engine()
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        )
        return list(result.scalars().all())

async def save_message(session_id: str, role: str, content: str):
    if not session_id or not role or not content:
        raise ValueError("session_id, role, and content are required")
    if role not in ["user", "assistant"]:
        raise ValueError("role must be 'user' or 'assistant'")
    await init_db()
    engine = get_engine()
    encrypted_content = encrypt(content)
    async with AsyncSession(engine) as session:
        if not await session.get(ChatSession, session_id):
            raise ValueError(f"Session {session_id} does not exist")
        session.add(ChatMessage(session_id=session_id, role=role, content=encrypted_content))
        await session.commit()

async def load_history(session_id: str) -> List[Tuple[str, str]]:
    if not session_id:
        return []
    await init_db()
    engine = get_engine()
    async with AsyncSession(engine) as session:
        if not await session.get(ChatSession, session_id):
            return []
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        )
        messages = result.scalars().all()
    return [(msg.role, decrypt(msg.content)) for msg in messages]

async def delete_session(session_id: str):
    if not session_id:
        raise ValueError("session_id is required")
    await init_db()
    engine = get_engine()
    async with AsyncSession(engine) as session:
        existing_session = await session.get(ChatSession, session_id)
        if existing_session:
            await session.delete(existing_session)
            await session.commit()

async def get_session_by_id(session_id: str) -> Optional[ChatSession]:
    if not session_id:
        return None
    await init_db()
    engine = get_engine()
    async with AsyncSession(engine) as session:
        return await session.get(ChatSession, session_id)

async def cleanup_empty_sessions():
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

async def get_database_stats():
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
