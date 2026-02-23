from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Language(str, Enum):
    FR = "fr"
    EN = "en"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# ========== REQUÊTES ==========

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, example="Qu'est-ce que CitizenLab ?")
    session_id: Optional[str] = Field(None, example="user-123")
    language: Language = Field(Language.FR)
    country_filter: Optional[str] = Field(None, example="senegal")


# ========== RÉPONSES ==========

class Source(BaseModel):
    content: str
    source_file: str
    relevance_score: float


class MessageHistory(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime


class ChatResponse(BaseModel):
    session_id: str
    response: str
    language: Language
    sources: List[Source] = []
    tokens_used: Optional[int] = None
    model: str


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    last_activity: datetime
    history: List[MessageHistory]


class HealthResponse(BaseModel):
    status: str
    version: str
    knowledge_base_loaded: bool
    total_chunks: int
    environment: str


class AdminStatsResponse(BaseModel):
    total_sessions: int
    active_sessions: int
    total_messages: int
    knowledge_base_stats: dict