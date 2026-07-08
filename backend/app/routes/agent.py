from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agent.loop import AgentSession
from ..agent.session_manager import manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    message: str
    context_files: list[str] = []


class ApprovalRequest(BaseModel):
    id: str
    approved: bool


class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    busy: bool


class AgentResponse(BaseModel):
    events: list[dict[str, Any]]
    busy: bool


class ActivityEntry(BaseModel):
    timestamp: Optional[str] = None
    category: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class ActivityHistory(BaseModel):
    session_id: str
    title: str
    events: list[dict[str, Any]]
    prompts: list[ActivityEntry]
    tool_calls: list[ActivityEntry]
    approvals: list[ActivityEntry]
    file_changes: list[ActivityEntry]
    errors: list[ActivityEntry]


def _get_session(session_id: str) -> AgentSession:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def _continue(session: AgentSession) -> AgentResponse:
    """Wait until the turn finishes or pauses on an approval, then return new events."""
    await session.wait_for_pause()
    return AgentResponse(events=session.drain(), busy=session.busy)


# --------------------------------------------------------------- session CRUD


@router.get("", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    return [SessionInfo(**s.info()) for s in manager.list()]


@router.post("", response_model=SessionInfo)
async def create_session(body: CreateSessionRequest = CreateSessionRequest()) -> SessionInfo:
    session = await manager.create(title=(body.title or "").strip() or None)
    return SessionInfo(**session.info())


@router.patch("/{session_id}", response_model=SessionInfo)
async def rename_session(session_id: str, body: RenameSessionRequest) -> SessionInfo:
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be empty")
    _get_session(session_id)
    session = await manager.rename(session_id, title[:255])
    return SessionInfo(**session.info())


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    _get_session(session_id)
    try:
        await manager.delete(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True}


# ----------------------------------------------------------- agent operations


@router.get("/{session_id}/state", response_model=AgentResponse)
async def get_state(session_id: str) -> AgentResponse:
    """Full transcript — lets the UI rebuild after a reload or session switch."""
    session = _get_session(session_id)
    return AgentResponse(events=session.full_state(), busy=session.busy)


@router.get("/{session_id}/activity", response_model=ActivityHistory)
async def get_activity(session_id: str) -> ActivityHistory:
    """Categorized activity history for audit/history views."""
    session = _get_session(session_id)
    prompts: list[ActivityEntry] = []
    tool_calls: list[ActivityEntry] = []
    approvals: list[ActivityEntry] = []
    file_changes: list[ActivityEntry] = []
    errors: list[ActivityEntry] = []

    for event in session.events:
        entry = _activity_entry(event)
        if entry is None:
            continue
        if entry.category == "prompt":
            prompts.append(entry)
        elif entry.category == "tool":
            tool_calls.append(entry)
        elif entry.category == "approval":
            approvals.append(entry)
        elif entry.category == "file":
            file_changes.append(entry)
        elif entry.category == "error":
            errors.append(entry)

    return ActivityHistory(
        session_id=session.id,
        title=session.title,
        events=list(session.events),
        prompts=prompts,
        tool_calls=tool_calls,
        approvals=approvals,
        file_changes=file_changes,
        errors=errors,
    )


@router.post("/{session_id}/continue", response_model=AgentResponse)
async def continue_turn(session_id: str) -> AgentResponse:
    """Long-poll for the next pause point (used after a reload mid-turn)."""
    return await _continue(_get_session(session_id))


@router.post("/{session_id}/chat", response_model=AgentResponse)
async def chat(session_id: str, body: ChatRequest) -> AgentResponse:
    session = _get_session(session_id)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")
    if session.busy:
        raise HTTPException(status_code=409, detail="Agent is busy with the current turn")
    session.start_turn(message, context_files=body.context_files)
    return await _continue(session)


@router.post("/{session_id}/approval", response_model=AgentResponse)
async def approval(session_id: str, body: ApprovalRequest) -> AgentResponse:
    session = _get_session(session_id)
    session.resolve_approval(body.id, body.approved)
    return await _continue(session)


@router.post("/{session_id}/stop", response_model=AgentResponse)
async def stop(session_id: str) -> AgentResponse:
    session = _get_session(session_id)
    session.request_stop()
    return await _continue(session)


@router.post("/{session_id}/reset", response_model=AgentResponse)
async def reset(session_id: str) -> AgentResponse:
    """Clear this session's transcript (keeps the session itself)."""
    session = _get_session(session_id)
    try:
        session.reset()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    await session.persist()
    return AgentResponse(events=[], busy=False)


def _activity_entry(event: dict[str, Any]) -> Optional[ActivityEntry]:
    timestamp = event.get("timestamp")
    if event.get("type") == "user_message":
        content = str(event.get("content") or "")
        return ActivityEntry(
            timestamp=timestamp,
            category="prompt",
            summary=content[:160],
            data={"context_files": event.get("context_files") or []},
        )
    if event.get("type") == "tool_call":
        name = str(event.get("name") or "tool")
        return ActivityEntry(
            timestamp=timestamp,
            category="tool",
            summary=f"Called {name}",
            data={"id": event.get("id"), "name": name, "args": event.get("args") or {}},
        )
    if event.get("type") == "approval_request":
        name = str(event.get("name") or "approval")
        target = event.get("path") or event.get("command") or ""
        return ActivityEntry(
            timestamp=timestamp,
            category="approval",
            summary=f"Requested approval for {name}",
            data={"id": event.get("id"), "name": name, "target": target},
        )
    if event.get("type") == "approval_resolved":
        approved = bool(event.get("approved"))
        return ActivityEntry(
            timestamp=timestamp,
            category="approval",
            summary="Approved request" if approved else "Rejected request",
            data={"id": event.get("id"), "approved": approved},
        )
    if event.get("type") == "files_changed":
        paths = event.get("paths") or []
        return ActivityEntry(
            timestamp=timestamp,
            category="file",
            summary=f"Changed {len(paths)} file(s)",
            data={"paths": paths},
        )
    if event.get("type") == "error":
        return ActivityEntry(
            timestamp=timestamp,
            category="error",
            summary=str(event.get("message") or "Agent error"),
            data={},
        )
    return None
