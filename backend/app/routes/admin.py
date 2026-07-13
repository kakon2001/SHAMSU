from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter

from ..agent.session_manager import manager

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/overview")
async def admin_overview() -> dict[str, Any]:
    totals: Counter[str] = Counter()
    sessions = []
    recent_events: list[dict[str, Any]] = []

    for session in manager.list():
        counts: Counter[str] = Counter()
        for event in session.events:
            kind = str(event.get("type") or "unknown")
            counts[kind] += 1
            totals[kind] += 1
            if kind in {"user_message", "approval_request", "approval_resolved", "files_changed", "error"}:
                recent_events.append(
                    {
                        "session_id": session.id,
                        "session_title": session.title,
                        "type": kind,
                        "timestamp": event.get("timestamp"),
                        "summary": _summary(event),
                    }
                )
        sessions.append(
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "busy": session.busy,
                "counts": dict(counts),
            }
        )

    recent_events.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return {
        "totals": dict(totals),
        "session_count": len(sessions),
        "sessions": sessions[:25],
        "recent_events": recent_events[:30],
    }


def _summary(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind == "user_message":
        return str(event.get("content") or "")[:180]
    if kind == "approval_request":
        name = str(event.get("name") or "approval")
        target = event.get("path") or event.get("command") or ""
        return f"Requested {name}: {target}"[:180]
    if kind == "approval_resolved":
        return "Approved request" if event.get("approved") else "Rejected request"
    if kind == "files_changed":
        paths = event.get("paths") or []
        return f"Changed {len(paths)} file(s)"
    if kind == "error":
        return str(event.get("message") or "Error")[:180]
    return str(kind or "event")
