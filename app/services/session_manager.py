import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages: List[Message] = field(default_factory=list)
    total_messages: int = 0


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.utcnow()
            return session
        new_id = session_id or str(uuid.uuid4())
        session = Session(session_id=new_id)
        self._sessions[new_id] = session
        logger.info(f"📝 Nouvelle session : {new_id}")
        return session

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self.get_or_create_session(session_id)
        session = self._sessions[session_id]
        session.messages.append(Message(role=role, content=content))
        session.total_messages += 1
        session.last_activity = datetime.utcnow()
        max_msgs = settings.MAX_HISTORY_LENGTH
        if len(session.messages) > max_msgs:
            session.messages = session.messages[-max_msgs:]

    def get_history(self, session_id: str) -> List[dict]:
        if session_id not in self._sessions:
            return []
        session = self._sessions[session_id]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages[:-1]
        ]

    def get_session_info(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        ttl = timedelta(minutes=settings.SESSION_TTL_MINUTES)
        now = datetime.utcnow()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_activity > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def get_stats(self) -> dict:
        now = datetime.utcnow()
        ttl = timedelta(minutes=settings.SESSION_TTL_MINUTES)
        active = sum(1 for s in self._sessions.values() if now - s.last_activity < ttl)
        total_msgs = sum(s.total_messages for s in self._sessions.values())
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active,
            "total_messages": total_msgs,
        }


session_manager = SessionManager()