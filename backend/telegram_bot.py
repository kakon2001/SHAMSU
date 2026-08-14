"""Telegram bridge for SHAMSU.

Run this after the SHAMSU backend is running:
    python telegram_bot.py

Commands in Telegram:
    /start              show help
    /new Project name   create a private SHAMSU project for this Telegram user
    /projects           list only this Telegram user's projects
    /use <project_id>   select one of this user's projects
    /history            show selected project history
    /stop               stop a busy selected project
    attach PDF/DOCX/TXT upload and index a PRD/document
    any text            send prompt to the selected project
"""

from __future__ import annotations

import base64
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
API_BASE = os.getenv("SHAMSU_API_BASE", "http://127.0.0.1:8080").rstrip("/")
BRIDGE_SECRET = os.getenv("TELEGRAM_BRIDGE_SECRET", "").strip()
POLL_INTERVAL = float(os.getenv("TELEGRAM_POLL_INTERVAL_SECONDS", "1.0"))
TELEGRAM_API = f"https://api.telegram.org/bot{TOKEN}"
_SELECTED_PROJECT: dict[str, str] = {}
_PENDING_APPROVAL: dict[str, dict[str, str]] = {}


def telegram_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in backend/.env")
    data = urllib.parse.urlencode(params or {}).encode("utf-8")
    req = urllib.request.Request(f"{TELEGRAM_API}/{method}", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def telegram_download_file(file_id: str) -> bytes:
    info = telegram_request("getFile", {"file_id": file_id})
    file_path = (info.get("result") or {}).get("file_path")
    if not file_path:
        raise RuntimeError("Telegram did not return a downloadable file path")
    url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def shamsu_request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    if not BRIDGE_SECRET or BRIDGE_SECRET == "change-this-telegram-bridge-secret":
        raise RuntimeError("Set TELEGRAM_BRIDGE_SECRET in backend/.env to a private value")
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        API_BASE + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Telegram-Bridge-Secret": BRIDGE_SECRET,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SHAMSU API error {exc.code}: {detail}") from exc


def send_message(chat_id: int, text: str) -> None:
    telegram_request("sendMessage", {"chat_id": chat_id, "text": text[:3900]})


def create_project(user_id: str, title: str) -> dict[str, Any]:
    project = shamsu_request("POST", "/api/telegram/projects", {"telegram_user_id": user_id, "title": title})
    _SELECTED_PROJECT[user_id] = project["id"]
    return project


def list_projects(user_id: str) -> list[dict[str, Any]]:
    return shamsu_request("GET", f"/api/telegram/users/{urllib.parse.quote(user_id)}/projects")


def project_history(user_id: str, project_id: str) -> str:
    query = urllib.parse.urlencode({"telegram_user_id": user_id})
    result = shamsu_request("GET", f"/api/telegram/projects/{project_id}/history?{query}")
    return result.get("summary") or "No project history yet."



def upload_document(user_id: str, project_id: str, filename: str, data: bytes) -> str:
    result = shamsu_request(
        "POST",
        f"/api/telegram/projects/{project_id}/document",
        {
            "telegram_user_id": user_id,
            "filename": filename,
            "data_base64": base64.b64encode(data).decode("ascii"),
        },
    )
    return result.get("reply") or f"Uploaded and indexed {filename}."


def approve_project(user_id: str, project_id: str, approval_id: str, approved: bool) -> dict[str, Any]:
    return shamsu_request(
        "POST",
        f"/api/telegram/projects/{project_id}/approval",
        {"telegram_user_id": user_id, "approval_id": approval_id, "approved": approved},
    )


def remember_pending_approval(user_id: str, project_id: str, response: dict[str, Any]) -> None:
    for event in reversed(response.get("events") or []):
        if event.get("type") == "approval_request" and event.get("id"):
            _PENDING_APPROVAL[user_id] = {"project_id": project_id, "approval_id": str(event["id"])}
            return
    if not response.get("busy"):
        _PENDING_APPROVAL.pop(user_id, None)


def handle_approval_reply(chat_id: int, user_id: str, text: str) -> bool:
    normalized = text.strip().lower()
    if normalized not in {"yes", "y", "approve", "approved", "ok approve", "no", "n", "reject", "rejected", "deny"}:
        return False
    pending = _PENDING_APPROVAL.get(user_id)
    if not pending:
        return False
    approved = normalized in {"yes", "y", "approve", "approved", "ok approve"}
    response = approve_project(user_id, pending["project_id"], pending["approval_id"], approved)
    remember_pending_approval(user_id, pending["project_id"], response)
    if not response.get("busy"):
        _PENDING_APPROVAL.pop(user_id, None)
    send_message(chat_id, response.get("reply") or ("Approved." if approved else "Rejected."))
    return True


def stop_selected_project(user_id: str, project_id: str) -> str:
    query = urllib.parse.urlencode({"telegram_user_id": user_id})
    result = shamsu_request("POST", f"/api/telegram/projects/{project_id}/stop?{query}")
    return result.get("message") or "Stop command sent."
def selected_or_latest_project(user_id: str) -> dict[str, Any] | None:
    projects = list_projects(user_id)
    if not projects:
        return None
    selected = _SELECTED_PROJECT.get(user_id)
    for project in projects:
        if project["id"] == selected:
            return project
    _SELECTED_PROJECT[user_id] = projects[0]["id"]
    return projects[0]


def handle_text(chat_id: int, user_id: str, text: str) -> None:
    text = text.strip()
    if handle_approval_reply(chat_id, user_id, text):
        return
    if text in {"/start", "help", "/help"}:
        send_message(chat_id, "SHAMSU Telegram bridge\n/new Project name - create private project\n/projects - list your projects\n/use <project_id> - select project\n/history - show selected project history\n/stop - stop a busy project\nAttach PDF/DOCX/TXT/MD/PRD files to index them.\nThen send any prompt to the selected project.")
        return
    if text.startswith("/new"):
        title = text.removeprefix("/new").strip() or "Telegram project"
        project = create_project(user_id, title)
        send_message(chat_id, f"Created private SHAMSU project:\n{project['title']}\nID: {project['id']}")
        return
    if text in {"/projects", "/project"}:
        projects = list_projects(user_id)
        if not projects:
            send_message(chat_id, "No SHAMSU projects yet. Send /new Project name")
            return
        lines = ["Your SHAMSU projects:"]
        for project in projects:
            active = " *selected*" if _SELECTED_PROJECT.get(user_id) == project["id"] else ""
            lines.append(f"- {project['title']}\n  {project['id']}{active}")
        send_message(chat_id, "\n".join(lines))
        return
    if text == "/history":
        project = selected_or_latest_project(user_id)
        if project is None:
            send_message(chat_id, "No SHAMSU project selected yet. Send /new Project name first.")
            return
        send_message(chat_id, project_history(user_id, project["id"]))
        return
    if text == "/stop":
        project = selected_or_latest_project(user_id)
        if project is None:
            send_message(chat_id, "No SHAMSU project selected yet. Send /new Project name first.")
            return
        _PENDING_APPROVAL.pop(user_id, None)
        send_message(chat_id, stop_selected_project(user_id, project["id"]))
        return
    if text.startswith("/use "):
        project_id = text.split(maxsplit=1)[1].strip()
        projects = list_projects(user_id)
        if not any(project["id"] == project_id for project in projects):
            send_message(chat_id, "That project was not found in your Telegram account.")
            return
        _SELECTED_PROJECT[user_id] = project_id
        send_message(chat_id, f"Selected SHAMSU project {project_id}")
        return
    if text.startswith("/"):
        send_message(chat_id, "Unknown command. Use /start, /new, /projects, /use, /history, or /stop.")
        return

    project = selected_or_latest_project(user_id)
    if project is None:
        project = create_project(user_id, "Telegram project")
        send_message(chat_id, f"Created your first private SHAMSU project: {project['title']}")
    response = shamsu_request(
        "POST",
        f"/api/telegram/projects/{project['id']}/message",
        {"telegram_user_id": user_id, "message": text},
    )
    remember_pending_approval(user_id, project["id"], response)
    send_message(chat_id, response.get("reply") or "SHAMSU finished, but no text reply was produced. Send /history to see project history.")
def handle_document(chat_id: int, user_id: str, document: dict[str, Any]) -> None:
    project = selected_or_latest_project(user_id)
    if project is None:
        project = create_project(user_id, "Telegram document project")
        send_message(chat_id, f"Created your first private SHAMSU project: {project['title']}")

    filename = document.get("file_name") or "telegram-upload.txt"
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".docx", ".txt", ".md", ".prd", ".csv", ".json", ".py", ".js", ".html", ".css", ".log"}
    if suffix not in allowed:
        send_message(chat_id, "Unsupported file type. Send PDF, DOCX, TXT, MD, PRD, or text/code files.")
        return
    file_size = int(document.get("file_size") or 0)
    if file_size and file_size > 10 * 1024 * 1024:
        send_message(chat_id, "This file is larger than SHAMSU's 10 MB upload limit.")
        return
    file_id = document.get("file_id")
    if not file_id:
        send_message(chat_id, "Telegram did not include a file id for this document.")
        return
    data = telegram_download_file(str(file_id))
    send_message(chat_id, upload_document(user_id, project["id"], filename, data))


def handle_document_safe(chat_id: int, user_id: str, document: dict[str, Any]) -> None:
    print(f"Telegram document from {user_id}: {document.get('file_name') or document.get('file_id')}")
    try:
        handle_document(chat_id, user_id, document)
    except Exception as exc:
        print(f"Telegram document failed for {user_id}: {exc}")
        try:
            send_message(chat_id, f"SHAMSU Telegram upload error: {exc}\nCheck backend and supported file type, then try again.")
        except Exception as send_exc:
            print(f"Could not send Telegram upload error message: {send_exc}")


def handle_text_safe(chat_id: int, user_id: str, text: str) -> None:
    print(f"Telegram message from {user_id}: {text}")
    try:
        handle_text(chat_id, user_id, text)
    except Exception as exc:
        print(f"Telegram command failed for {user_id}: {exc}")
        try:
            send_message(chat_id, f"SHAMSU Telegram error: {exc}\nCheck that backend is running, then restart python telegram_bot.py.")
        except Exception as send_exc:
            print(f"Could not send Telegram error message: {send_exc}")


def main() -> None:
    offset = 0
    print("SHAMSU Telegram bridge is running. Press Ctrl+C to stop.")
    while True:
        try:
            result = telegram_request("getUpdates", {"offset": offset, "timeout": 30})
            for update in result.get("result", []):
                offset = max(offset, int(update["update_id"]) + 1)
                message = update.get("message") or update.get("edited_message") or {}
                text = message.get("text")
                document = message.get("document")
                chat = message.get("chat") or {}
                user = message.get("from") or {}
                if "id" not in chat or "id" not in user:
                    continue
                if document:
                    handle_document_safe(int(chat["id"]), str(user["id"]), document)
                    continue
                if not text:
                    continue
                handle_text_safe(int(chat["id"]), str(user["id"]), text)
        except (TimeoutError, socket.timeout) as exc:
            print(f"Telegram polling timeout, continuing: {exc}")
            continue
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                print("Telegram polling timeout, continuing.")
                continue
            print(f"Telegram network error: {exc}")
            time.sleep(3)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error: {exc.code} {detail}")
            time.sleep(3)
        except Exception as exc:
            print(f"Telegram bridge error: {exc}")
            time.sleep(3)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()