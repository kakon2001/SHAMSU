from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends

from .. import db
from ..agent.session_manager import manager
from .auth import required_user

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/overview")
async def admin_overview(user: dict[str, Any] = Depends(required_user)) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    sessions = []
    recent_events: list[dict[str, Any]] = []
    generated_files: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []

    for session in manager.list(owner_id=str(user["id"])):
        counts: Counter[str] = Counter()
        changed_paths: set[str] = set()
        verification_events = 0
        failed_verification_events = 0
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
            if kind == "files_changed":
                for path in event.get("paths") or []:
                    rel = str(path)
                    changed_paths.add(rel)
                    item = {
                        "path": rel,
                        "session_id": session.id,
                        "session_title": session.title,
                        "timestamp": event.get("timestamp"),
                    }
                    generated_files.append(item)
                    if rel.startswith("uploads/"):
                        uploads.append(item)
            if "verify" in kind or kind in {"verification", "tests_run"}:
                verification_events += 1
                if event.get("ok") is False or event.get("status") in {"error", "failed"}:
                    failed_verification_events += 1
        sessions.append(
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "busy": session.busy,
                "counts": dict(counts),
                "generated_files": sorted(changed_paths)[:20],
                "verification_status": _session_verification_status(verification_events, failed_verification_events),
            }
        )

    recent_events.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    generated_files.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    uploads.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    telegram_stats = await db.telegram_project_stats()
    return {
        "totals": dict(totals),
        "session_count": len(sessions),
        "sessions": sessions[:25],
        "recent_events": recent_events[:30],
        "persistent_dashboard": {
            "local_projects": _local_projects_from_files(generated_files),
            "telegram_projects": telegram_stats,
            "uploads": uploads[:20],
            "generated_files": generated_files[:40],
            "verification_status": _overall_verification_status(sessions),
        },
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


def _session_verification_status(total: int, failed: int) -> str:
    if failed:
        return "needs repair"
    if total:
        return "verified"
    return "not verified"


def _overall_verification_status(sessions: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(session.get("verification_status") or "not verified") for session in sessions)
    return dict(counts)


def _local_projects_from_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projects: dict[str, dict[str, Any]] = {}
    for item in files:
        path = str(item.get("path") or "")
        if not path or path.startswith("uploads/"):
            continue
        project = path.split("/", 1)[0] if "/" in path else path
        entry = projects.setdefault(
            project,
            {
                "name": project.replace("_", " ").replace(".html", "").title(),
                "path": project,
                "files": 0,
                "generated_files": [],
                "previewable": False,
                "verification_status": "needs smoke test",
            },
        )
        entry["files"] += 1
        entry["generated_files"].append(path)
        if path.endswith("index.html") or path.endswith(".html"):
            entry["previewable"] = True
        if path.endswith("smoke_test.py") or "/tests/" in path:
            entry["verification_status"] = "has smoke test"
    return list(projects.values())[:40]