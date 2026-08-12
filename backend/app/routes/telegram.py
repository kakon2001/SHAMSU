from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from .. import db
from ..agent.session_manager import manager
from ..config import settings

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class TelegramProjectRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    title: str = "Telegram project"


class TelegramMessageRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TelegramProject(BaseModel):
    id: str
    telegram_user_id: str
    title: str
    session_id: str
    created_at: str
    updated_at: str


class TelegramMessageResponse(BaseModel):
    project: TelegramProject
    reply: str
    busy: bool
    events: list[dict[str, Any]]


class TelegramHistoryResponse(BaseModel):
    project: TelegramProject
    events: list[dict[str, Any]]
    summary: str


class TelegramStopResponse(BaseModel):
    project: TelegramProject
    stopped: bool
    message: str


def _require_bridge_secret(x_telegram_bridge_secret: str | None) -> None:
    expected = settings.telegram_bridge_secret.strip()
    if not expected or expected == "change-this-telegram-bridge-secret":
        raise HTTPException(status_code=503, detail="Telegram bridge secret is not configured")
    if x_telegram_bridge_secret != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Telegram bridge secret")


def _owner_id(telegram_user_id: str) -> str:
    return f"telegram:{telegram_user_id.strip()}"


def _project_response(project: dict[str, Any]) -> TelegramProject:
    return TelegramProject(
        id=project["id"],
        telegram_user_id=project["telegram_user_id"],
        title=project["title"],
        session_id=project["session_id"],
        created_at=project["created_at"].isoformat() if hasattr(project["created_at"], "isoformat") else str(project["created_at"]),
        updated_at=project["updated_at"].isoformat() if hasattr(project["updated_at"], "isoformat") else str(project["updated_at"]),
    )


@router.post("/projects", response_model=TelegramProject)
async def create_project(
    body: TelegramProjectRequest,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> TelegramProject:
    _require_bridge_secret(x_telegram_bridge_secret)
    telegram_user_id = body.telegram_user_id.strip()
    title = body.title.strip()[:80] or "Telegram project"
    session = await manager.create(title=title, owner_id=_owner_id(telegram_user_id))
    project = await db.create_telegram_project(telegram_user_id=telegram_user_id, title=title, session_id=session.id)
    return _project_response(project)


@router.get("/users/{telegram_user_id}/projects", response_model=list[TelegramProject])
async def list_projects(
    telegram_user_id: str,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> list[TelegramProject]:
    _require_bridge_secret(x_telegram_bridge_secret)
    projects = await db.list_telegram_projects(telegram_user_id.strip())
    return [_project_response(project) for project in projects]


@router.post("/projects/{project_id}/message", response_model=TelegramMessageResponse)
async def send_project_message(
    project_id: str,
    body: TelegramMessageRequest,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> TelegramMessageResponse:
    _require_bridge_secret(x_telegram_bridge_secret)
    telegram_user_id = body.telegram_user_id.strip()
    project = await db.get_telegram_project(project_id, telegram_user_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Telegram project not found for this user")

    session = manager.get(project["session_id"])
    if session is None or session.owner_id != _owner_id(telegram_user_id):
        raise HTTPException(status_code=404, detail="Telegram project session not found")
    if session.busy:
        raise HTTPException(status_code=409, detail="This project is already running a SHAMSU task")

    message = body.message.strip()
    if _is_simple_telegram_chat(message):
        reply = _simple_telegram_reply(message)
        session._emit({"type": "user_message", "content": message, "context_files": [], "source": "telegram"})
        session._emit({"type": "assistant_message", "content": reply, "source": "telegram"})
        session._emit({"type": "turn_end", "source": "telegram"})
        await session.persist()
        events = session.drain()
    else:
        session.start_turn(message)
        await session.wait_for_pause()
        events = session.drain()
        reply = _last_assistant_reply(events) or _history_summary(session.full_state()[-8:]) or "SHAMSU recorded the prompt. Send /history to see this project history."
    await db.touch_telegram_project(project_id)
    refreshed = await db.get_telegram_project(project_id, telegram_user_id) or project
    return TelegramMessageResponse(project=_project_response(refreshed), reply=reply, busy=session.busy, events=events)


@router.get("/projects/{project_id}/history", response_model=TelegramHistoryResponse)
async def get_project_history(
    project_id: str,
    telegram_user_id: str,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> TelegramHistoryResponse:
    _require_bridge_secret(x_telegram_bridge_secret)
    telegram_user_id = telegram_user_id.strip()
    project = await db.get_telegram_project(project_id, telegram_user_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Telegram project not found for this user")
    session = manager.get(project["session_id"])
    if session is None or session.owner_id != _owner_id(telegram_user_id):
        return TelegramHistoryResponse(project=_project_response(project), events=[], summary="Project record exists, but its live SHAMSU session is not loaded. Restart the backend and Telegram bot, then send /projects and /history again.")
    events = session.full_state()[-20:]
    return TelegramHistoryResponse(project=_project_response(project), events=events, summary=_history_summary(events))


@router.post("/projects/{project_id}/stop", response_model=TelegramStopResponse)
async def stop_project_turn(
    project_id: str,
    telegram_user_id: str,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> TelegramStopResponse:
    _require_bridge_secret(x_telegram_bridge_secret)
    telegram_user_id = telegram_user_id.strip()
    project = await db.get_telegram_project(project_id, telegram_user_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Telegram project not found for this user")
    session = manager.get(project["session_id"])
    if session is None or session.owner_id != _owner_id(telegram_user_id):
        return TelegramStopResponse(project=_project_response(project), stopped=False, message="Project session is not loaded. Restart backend and Telegram bot.")
    if session.busy:
        session.request_stop()
        await session.wait_for_pause()
        await session.persist()
        return TelegramStopResponse(project=_project_response(project), stopped=True, message="Stopped the running SHAMSU task for this project.")
    return TelegramStopResponse(project=_project_response(project), stopped=False, message="No running SHAMSU task in this project.")

def _last_assistant_reply(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "assistant_message" and event.get("content"):
            return str(event["content"])
    return ""


def _history_summary(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "user_message":
            lines.append("You: " + str(event.get("content") or "")[:160])
        elif kind == "assistant_message":
            lines.append("SHAMSU: " + str(event.get("content") or "")[:240])
        elif kind == "tool_call":
            lines.append("Tool: " + str(event.get("name") or "tool"))
        elif kind == "approval_request":
            lines.append("Approval requested")
    return "\n".join(lines[-12:]) or "No project history yet."

def _is_simple_telegram_chat(message: str) -> bool:
    lower = message.lower().strip()
    coding_terms = [
        "create file", "write file", "edit file", "delete file", "run ", "build ", "make ",
        "generate", "implement", "fix", "bug", "error", "workspace", "project", "prd",
        "html", "python", "javascript", "fastapi", "database", "test", "deploy",
    ]
    return not any(term in lower for term in coding_terms)


def _simple_telegram_reply(message: str) -> str:
    lower = message.lower().strip()
    if "hello" in lower or "hi" == lower:
        return "Hello!\nI am SHAMSU, your local coding agent connected through Telegram."
    if "what" in lower and "shamsu" in lower and "do" in lower:
        return "SHAMSU can answer project questions, create private Telegram projects, keep per-user history, and run coding workflows through your local backend."
    if "explain" in lower and "shamsu" in lower:
        return "SHAMSU is a local coding-agent system. It can manage chats/projects, read workspace files, build small apps, run CLI workflows, and keep history per user."
    return "I received your message in this private SHAMSU Telegram project. For coding work, ask me to build, edit, fix, or inspect a project."
