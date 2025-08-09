from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

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
