"""Telegram bridge for SHAMSU.

Run this after the SHAMSU backend is running:
    python telegram_bot.py

Commands in Telegram:
    /start              show help
    /new Project name   create a private SHAMSU project for this Telegram user
    /projects           list only this Telegram user's projects
    /use <project_id>   select one of this user's projects
    any text            send prompt to the selected project
"""

from __future__ import annotations

import json
import os
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


def telegram_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in backend/.env")
    data = urllib.parse.urlencode(params or {}).encode("utf-8")
    req = urllib.request.Request(f"{TELEGRAM_API}/{method}", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


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
    with urllib.request.urlopen(req, timeout=180) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else None


def send_message(chat_id: int, text: str) -> None:
    telegram_request("sendMessage", {"chat_id": chat_id, "text": text[:3900]})


def create_project(user_id: str, title: str) -> dict[str, Any]:
    project = shamsu_request("POST", "/api/telegram/projects", {"telegram_user_id": user_id, "title": title})
    _SELECTED_PROJECT[user_id] = project["id"]
    return project


def list_projects(user_id: str) -> list[dict[str, Any]]:
    return shamsu_request("GET", f"/api/telegram/users/{urllib.parse.quote(user_id)}/projects")


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
    if text in {"/start", "help", "/help"}:
        send_message(chat_id, "SHAMSU Telegram bridge\n/new Project name - create private project\n/projects - list your projects\n/use <project_id> - select project\nThen send any prompt to the selected project.")
        return
    if text.startswith("/new"):
        title = text.removeprefix("/new").strip() or "Telegram project"
        project = create_project(user_id, title)
        send_message(chat_id, f"Created private SHAMSU project:\n{project['title']}\nID: {project['id']}")
        return
    if text == "/projects":
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
    if text.startswith("/use "):
        project_id = text.split(maxsplit=1)[1].strip()
        projects = list_projects(user_id)
        if not any(project["id"] == project_id for project in projects):
            send_message(chat_id, "That project was not found in your Telegram account.")
            return
        _SELECTED_PROJECT[user_id] = project_id
        send_message(chat_id, f"Selected SHAMSU project {project_id}")
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
    send_message(chat_id, response.get("reply") or "SHAMSU finished, but no text reply was produced.")


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
                chat = message.get("chat") or {}
                user = message.get("from") or {}
                if not text or "id" not in chat or "id" not in user:
                    continue
                handle_text(int(chat["id"]), str(user["id"]), text)
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