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

    session.start_turn(body.message.strip())
    await session.wait_for_pause()
    events = session.drain()
    await db.touch_telegram_project(project_id)
    refreshed = await db.get_telegram_project(project_id, telegram_user_id) or project
    reply = _last_assistant_reply(events) or "SHAMSU recorded the prompt. Check the project history for details."
    return TelegramMessageResponse(project=_project_response(refreshed), reply=reply, busy=session.busy, events=events)


def _last_assistant_reply(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "assistant_message" and event.get("content"):
            return str(event["content"])
    return ""