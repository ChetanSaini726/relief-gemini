from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class ChatSession(SQLModel, table=True):
    """Chat session model"""
    id: str = Field(primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    messages: List["ChatMessage"] = Relationship(back_populates="session", sa_relationship_kwargs={"cascade": "all, delete"})

class ChatMessage(SQLModel, table=True):
    """Chat message model with session support"""
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str
    content: bytes  # encrypted blob

    session: ChatSession = Relationship(back_populates="messages")
