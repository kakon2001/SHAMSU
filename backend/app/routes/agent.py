from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agent.loop import session

router = APIRouter(prefix="/api/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    context_files: list[str] = []


class ApprovalRequest(BaseModel):
    id: str
    approved: bool


class AgentResponse(BaseModel):
    events: list[dict[str, Any]]
    busy: bool


async def _continue() -> AgentResponse:
    """Wait until the turn finishes or pauses on an approval, then return new events."""
    await session.wait_for_pause()
    return AgentResponse(events=session.drain(), busy=session.busy)


@router.get("/state", response_model=AgentResponse)
async def get_state() -> AgentResponse:
    """Full transcript — lets the UI rebuild after a page reload."""
    return AgentResponse(events=session.full_state(), busy=session.busy)


@router.post("/continue", response_model=AgentResponse)
async def continue_turn() -> AgentResponse:
    """Long-poll for the next pause point (used after a reload mid-turn)."""
    return await _continue()


@router.post("/chat", response_model=AgentResponse)
async def chat(body: ChatRequest) -> AgentResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")
    if session.busy:
        raise HTTPException(status_code=409, detail="Agent is busy with the current turn")
    session.start_turn(message, context_files=body.context_files)
    return await _continue()


@router.post("/approval", response_model=AgentResponse)
async def approval(body: ApprovalRequest) -> AgentResponse:
    session.resolve_approval(body.id, body.approved)
    return await _continue()


@router.post("/stop", response_model=AgentResponse)
async def stop() -> AgentResponse:
    session.request_stop()
    return await _continue()


@router.post("/reset", response_model=AgentResponse)
async def reset() -> AgentResponse:
    try:
        session.reset()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return AgentResponse(events=[], busy=False)
