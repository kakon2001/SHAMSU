"""Registry of chat sessions.

Holds every AgentSession in memory (fine for a single-user tool) and mirrors
creations/deletions to MySQL. Stored sessions are hydrated once at startup;
per-turn persistence is handled by AgentSession itself.
"""

from typing import Optional

from .. import db
from .loop import DEFAULT_TITLE, AgentSession


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    async def load_from_db(self) -> None:
        for row in await db.load_sessions():
            session = AgentSession(
                session_id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                conversation=row["conversation"],
                events=row["events"],
            )
            self._sessions[session.id] = session

    def list(self) -> list[AgentSession]:
        return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def get(self, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get(session_id)

    async def create(self, title: Optional[str] = None) -> AgentSession:
        session = AgentSession(title=title or DEFAULT_TITLE)
        self._sessions[session.id] = session
        await session.persist()
        return session

    async def delete(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.busy:
            raise RuntimeError("Stop the current turn before deleting this session")
        del self._sessions[session_id]
        await db.delete_session(session_id)

    async def rename(self, session_id: str, title: str) -> AgentSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        session.title = title
        await session.persist()
        return session


manager = SessionManager()
