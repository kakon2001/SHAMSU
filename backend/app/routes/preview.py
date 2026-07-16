from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agent.tools import resolve_in_workspace
from ..config import settings

router = APIRouter(prefix="/api/preview", tags=["preview"])

_preview_process: Optional[subprocess.Popen[str]] = None
_preview_port: Optional[int] = None


class PreviewStartRequest(BaseModel):
    path: str = ""
    port: int = Field(default=9000, ge=1024, le=65535)


class PreviewState(BaseModel):
    running: bool
    managed: bool
    port: int
    url: str
    path: str
    message: str


@router.get("/status", response_model=PreviewState)
async def preview_status(port: int = 9000, path: str = "") -> PreviewState:
    return _state(port=port, path=path, message="Preview status checked.")


@router.post("/start", response_model=PreviewState)
async def start_preview(body: PreviewStartRequest) -> PreviewState:
    global _preview_process, _preview_port
    target_path = _validate_preview_path(body.path)
    port = body.port

    if _process_running(_preview_process):
        _preview_port = _preview_port or port
        return _state(port=_preview_port, path=target_path, message="Preview server is already running.")

    if _port_open("127.0.0.1", port):
        return _state(
            port=port,
            path=target_path,
            message="Port is already serving a preview. Reusing the existing server.",
            managed=False,
        )

    try:
        _preview_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(settings.workdir_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=_creationflags(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start preview server: {exc}")

    _preview_port = port
    return _state(port=port, path=target_path, message="Preview server started.")


@router.post("/stop", response_model=PreviewState)
async def stop_preview() -> PreviewState:
    global _preview_process, _preview_port
    port = _preview_port or 9000
    if _process_running(_preview_process):
        _preview_process.terminate()
        try:
            _preview_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _preview_process.kill()
            _preview_process.wait(timeout=5)
        _preview_process = None
        _preview_port = None
        return _state(port=port, path="", message="Managed preview server stopped.")
    _preview_process = None
    _preview_port = None
    return _state(port=port, path="", message="No managed preview server was running.")


def _validate_preview_path(path: str) -> str:
    cleaned = (path or "").strip()
    if not cleaned:
        return ""
    target = resolve_in_workspace(cleaned)
    if target.is_dir():
        return ""
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"'{cleaned}' does not exist in the workspace")
    return cleaned.replace("\\", "/")


def _state(port: int, path: str, message: str, managed: Optional[bool] = None) -> PreviewState:
    running = _process_running(_preview_process) or _port_open("127.0.0.1", port)
    is_managed = _process_running(_preview_process) if managed is None else managed
    url_path = f"/{path}" if path else "/"
    return PreviewState(
        running=running,
        managed=is_managed,
        port=port,
        path=path,
        url=f"http://127.0.0.1:{port}{url_path}",
        message=message,
    )


def _process_running(proc: Optional[subprocess.Popen[str]]) -> bool:
    return proc is not None and proc.poll() is None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _creationflags() -> int:
    if sys.platform.startswith("win"):
        return subprocess.CREATE_NO_WINDOW
    return 0
