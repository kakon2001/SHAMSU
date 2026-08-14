from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile
from io import BytesIO

from .. import db
from ..agent.session_manager import manager
from ..config import settings
from . import tasks, uploads

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class TelegramProjectRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    title: str = "Telegram project"


class TelegramMessageRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TelegramDocumentRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    data_base64: str = Field(min_length=1)


class TelegramApprovalRequest(BaseModel):
    telegram_user_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    approved: bool


class TelegramDocumentResponse(BaseModel):
    project: TelegramProject
    name: str
    path: str
    chars: int
    kind: str
    reply: str


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
    direct_reply = await _telegram_direct_task_reply(message)
    if direct_reply:
        reply = direct_reply
        session._emit({"type": "user_message", "content": message, "context_files": [], "source": "telegram"})
        session._emit({"type": "assistant_message", "content": reply, "source": "telegram"})
        session._emit({"type": "turn_end", "source": "telegram"})
        await session.persist()
        events = session.drain()
    elif _is_simple_telegram_chat(message):
        reply = _simple_telegram_reply(message)
        session._emit({"type": "user_message", "content": message, "context_files": [], "source": "telegram"})
        session._emit({"type": "assistant_message", "content": reply, "source": "telegram"})
        session._emit({"type": "turn_end", "source": "telegram"})
        await session.persist()
        events = session.drain()
    else:
        context_files, context_labels = _telegram_project_context_files(session)
        session.start_turn(message, context_files=context_files, context_file_labels=context_labels)
        await session.wait_for_pause()
        events = session.drain()
        reply = _reply_from_events(events) or _history_summary(session.full_state()[-8:]) or "SHAMSU recorded the prompt. Send /history to see this project history."
    await db.touch_telegram_project(project_id)
    refreshed = await db.get_telegram_project(project_id, telegram_user_id) or project
    return TelegramMessageResponse(project=_project_response(refreshed), reply=reply, busy=session.busy, events=events)




@router.post("/projects/{project_id}/approval", response_model=TelegramMessageResponse)
async def resolve_project_approval(
    project_id: str,
    body: TelegramApprovalRequest,
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
    if not session.busy:
        raise HTTPException(status_code=409, detail="No running approval is waiting in this project")

    session.resolve_approval(body.approval_id, body.approved)
    await session.wait_for_pause()
    events = session.drain()
    await db.touch_telegram_project(project_id)
    refreshed = await db.get_telegram_project(project_id, telegram_user_id) or project
    action = "approved" if body.approved else "rejected"
    reply = _reply_from_events(events) or f"Approval {action}. SHAMSU is continuing the project task."
    return TelegramMessageResponse(project=_project_response(refreshed), reply=reply, busy=session.busy, events=events)


@router.post("/projects/{project_id}/document", response_model=TelegramDocumentResponse)
async def upload_project_document(
    project_id: str,
    body: TelegramDocumentRequest,
    x_telegram_bridge_secret: str | None = Header(default=None),
) -> TelegramDocumentResponse:
    _require_bridge_secret(x_telegram_bridge_secret)
    telegram_user_id = body.telegram_user_id.strip()
    project = await db.get_telegram_project(project_id, telegram_user_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Telegram project not found for this user")
    session = manager.get(project["session_id"])
    if session is None or session.owner_id != _owner_id(telegram_user_id):
        raise HTTPException(status_code=404, detail="Telegram project session not found")

    try:
        raw = base64.b64decode(body.data_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Telegram file payload") from exc
    upload = UploadFile(filename=Path(body.filename).name, file=BytesIO(raw))
    result = await uploads.upload_context_file(upload)
    rel_path = str(result["path"])
    name = str(result["name"])
    reply = (
        f"Uploaded and indexed `{name}` for this Telegram project.\n"
        f"Workspace context: {rel_path}\n"
        f"Extracted {result['chars']} characters as {result['kind']}.\n\n"
        "Now ask: summarize this PRD, list requirements, or create a build roadmap."
    )
    session._emit({
        "type": "telegram_upload",
        "name": name,
        "path": rel_path,
        "kind": result["kind"],
        "chars": result["chars"],
        "summary": result["summary"],
        "source": "telegram",
    })
    session._emit({"type": "assistant_message", "content": reply, "source": "telegram"})
    await session.persist()
    await db.touch_telegram_project(project_id)
    refreshed = await db.get_telegram_project(project_id, telegram_user_id) or project
    return TelegramDocumentResponse(
        project=_project_response(refreshed),
        name=name,
        path=rel_path,
        chars=int(result["chars"]),
        kind=str(result["kind"]),
        reply=reply,
    )


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

def _reply_from_events(events: list[dict[str, Any]]) -> str:
    approval = _last_approval_request(events)
    if approval:
        return _format_approval_request(approval)
    return _last_assistant_reply(events)


def _last_assistant_reply(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "assistant_message" and event.get("content"):
            return str(event["content"])
    return ""


def _last_approval_request(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") == "approval_request" and event.get("id"):
            return event
    return None


def _format_approval_request(event: dict[str, Any]) -> str:
    name = str(event.get("name") or "approval")
    approval_id = str(event.get("id") or "")
    if event.get("command"):
        detail = f"Command: {event.get('command')}\nRisk: {event.get('risk') or 'unknown'} - {event.get('risk_reason') or 'review before approving'}"
    else:
        path = event.get("path") or "file"
        action = "creating" if event.get("is_new_file") else "writing"
        diff = str(event.get("diff") or "")[:1200]
        detail = f"Approve {action} `{path}`?\nDiff preview:\n{diff}"
    return (
        "Approval required in SHAMSU Telegram.\n"
        f"Tool: {name}\n"
        f"Approval ID: {approval_id}\n"
        f"{detail}\n\n"
        "Reply `yes` to approve or `no` to reject."
    )


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
        elif kind == "telegram_upload":
            lines.append("Upload: " + str(event.get("name") or "document") + " -> " + str(event.get("path") or ""))
        elif kind == "approval_request":
            lines.append("Approval requested")
    return "\n".join(lines[-12:]) or "No project history yet."

def _telegram_project_context_files(session: Any, limit: int = 5) -> tuple[list[str], dict[str, str]]:
    paths: list[str] = []
    labels: dict[str, str] = {}
    for event in reversed(session.full_state()):
        if event.get("type") != "telegram_upload":
            continue
        path = str(event.get("path") or "")
        name = str(event.get("name") or path)
        if path and path not in paths:
            paths.append(path)
            labels[path] = name
        if len(paths) >= limit:
            break
    paths.reverse()
    return paths, labels


def _is_simple_telegram_chat(message: str) -> bool:
    lower = message.lower().strip()
    coding_terms = [
        "create file", "create a file", "create new file", "write file", "write a file",
        "edit file", "delete file", "run ", "build ", "make ", "generate", "implement",
        "fix", "bug", "error", "workspace", "project", "prd", "html", "python",
        "javascript", "fastapi", "database", "test", "deploy", "code", "program",
        "script", "function", "class", "api", "app", "website",
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


async def _telegram_direct_task_reply(message: str) -> str:
    lower = message.lower().strip()
    if _wants_hello_world_code(lower):
        path = _write_workspace_text("telegram_hello_world.py", 'print("Hello, World!")\n')
        return (
            "Created `telegram_hello_world.py` in the SHAMSU workspace.\n\n"
            "```python\n"
            'print("Hello, World!")\n'
            "```\n\n"
            f"Workspace file: {path}"
        )

    file_match = re.search(
        r"(?:create|write|build)\s+(?:a\s+|new\s+)?file\s+(?:called|named)?\s*`?([A-Za-z0-9_.\\/-]+)`?",
        message,
        flags=re.IGNORECASE,
    )
    if file_match:
        filename = file_match.group(1).strip().strip("`.,")
        content = _starter_file_content(filename, message)
        path = _write_workspace_text(filename, content)
        return f"Created `{filename}` in the SHAMSU workspace.\n\nWorkspace file: {path}\n\n```{_fence_language(filename)}\n{content}```"

    if _looks_like_code_request(lower):
        code = _starter_code_for_prompt(lower, message)
        if code:
            filename, language, content = code
            path = _write_workspace_text(filename, content)
            return f"Created `{filename}` in the SHAMSU workspace.\n\n```{language}\n{content}```\nWorkspace file: {path}"
        return (
            "I understand this is a coding request, but I should not guess and create the wrong file.\n"
            "Please send: `create file <name.py> with code to <exact task>`"
        )
    if _looks_like_telegram_build_request(lower):
        result = await tasks.run_task(tasks.TaskRunRequest(prompt=message, preview=True, overwrite=True))
        created = "\n".join(f"- {name}" for name in result.created_files) or "- no file created"
        preview = f"\nPreview: {result.preview_url}" if result.preview_url else ""
        status = "passed" if result.ok else "needs follow-up"
        return f"Build status: {status}\nMode: {result.mode}\nCreated files:\n{created}{preview}"
    return ""


def _wants_hello_world_code(lower: str) -> bool:
    return "hello world" in lower and any(term in lower for term in ["code", "program", "script", "print", "write"])


def _extract_print_text(message: str) -> str:
    quoted = re.search(r'print\s+["?](.*?)["?]', message, flags=re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()
    says = re.search(r'(?:print|say|show)\s+(?:that\s+)?(?:will\s+)?(?:prints?\s+)?(.+)$', message, flags=re.IGNORECASE)
    if says and not re.search(r'\b(1 to 100|table|loop|factorial|calculator|fibonacci|prime|palindrome|odd|even)\b', says.group(1), flags=re.IGNORECASE):
        return says.group(1).strip().strip('.!?')
    return ""


def _looks_like_code_request(lower: str) -> bool:
    return any(
        term in lower
        for term in [
            "write me a code", "write code", "write a code", "build a code",
            "give me code", "code to", "program to", "script to", "make code",
            "for loop", "print 1 to", "print the table", "print table",
            "write me", "can you write", "need code", "make a program",
        ]
    )


def _starter_code_for_prompt(lower: str, original: str = "") -> tuple[str, str, str] | None:
    printable = _extract_print_text(original)
    if printable:
        escaped = printable.replace("\\", "\\\\").replace('"', '\\"')
        return (_preferred_filename(original, "telegram_code.py"), "python", f'print("{escaped}")\n')
    range_match = re.search(r'(?:print|show|display)\s+(\d+)\s*(?:to|-)\s*(\d+)', lower)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        step = 1 if end >= start else -1
        stop = end + step
        fallback = "telegram_loop_1_to_100.py" if (start, end) == (1, 100) else "telegram_number_loop.py"
        range_args = f"{start}, {stop}" if step == 1 else f"{start}, {stop}, {step}"
        return (_preferred_filename(original, fallback), "python", f"for number in range({range_args}):\n    print(number)\n")
    if "1 to 100" in lower or "1-100" in lower or "one to hundred" in lower or "one to 100" in lower:
        return (_preferred_filename(original, "telegram_loop_1_to_100.py"), "python", "for number in range(1, 101):\n    print(number)\n")
    if "for loop" in lower and ("table" in lower or "multiplication" in lower):
        return (
            _preferred_filename(original, "telegram_multiplication_table.py"),
            "python",
            "n = int(input('Enter a number: '))\n"
            "for i in range(1, 11):\n"
            "    print(f'{n} x {i} = {n * i}')\n",
        )
    if "for loop" in lower and "print" in lower:
        return (_preferred_filename(original, "telegram_for_loop.py"), "python", "for number in range(1, 11):\n    print(number)\n")
    if "fibonacci" in lower:
        return (_preferred_filename(original, "telegram_fibonacci.py"), "python", "n = int(input('How many terms? '))\na, b = 0, 1\nfor _ in range(n):\n    print(a)\n    a, b = b, a + b\n")
    if "prime" in lower:
        return (_preferred_filename(original, "telegram_prime.py"), "python", "n = int(input('Enter a number: '))\nif n > 1 and all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)):\n    print('Prime')\nelse:\n    print('Not prime')\n")
    if "even" in lower or "odd" in lower:
        return (_preferred_filename(original, "telegram_even_odd.py"), "python", "n = int(input('Enter a number: '))\nif n % 2 == 0:\n    print('Even')\nelse:\n    print('Odd')\n")
    if "palindrome" in lower:
        return (_preferred_filename(original, "telegram_palindrome.py"), "python", "text = input('Enter text: ')\ncleaned = text.replace(' ', '').lower()\nprint('Palindrome' if cleaned == cleaned[::-1] else 'Not palindrome')\n")
    if "factorial" in lower:
        return (
            _preferred_filename(original, "telegram_factorial.py"),
            "python",
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    if n in (0, 1):\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n\n"
            "print(factorial(5))\n",
        )
    if "calculator" in lower:
        return (
            _preferred_filename(original, "telegram_calculator.py"),
            "python",
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def subtract(a, b):\n"
            "    return a - b\n\n"
            "def multiply(a, b):\n"
            "    return a * b\n\n"
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('Cannot divide by zero')\n"
            "    return a / b\n\n"
            "print(add(2, 3))\n",
        )
    if "python" in lower or "print" in lower:
        return (_preferred_filename(original, "telegram_code.py"), "python", 'print("Hello from SHAMSU Telegram")\n')
    if "html" in lower or "website" in lower:
        return (
            _preferred_filename(original, "telegram_page.html"),
            "html",
            "<!doctype html>\n"
            "<html>\n"
            "<head><meta charset=\"utf-8\"><title>SHAMSU Page</title></head>\n"
            "<body><h1>Hello from SHAMSU Telegram</h1></body>\n"
            "</html>\n",
        )
    return None


def _preferred_filename(message: str, fallback: str) -> str:
    match = re.search(r'(?:file|named|called|save as)\s+`?([A-Za-z0-9_.\/-]+)`?', message, flags=re.IGNORECASE)
    if not match:
        return fallback
    filename = match.group(1).strip().strip("`.,")
    if Path(filename).suffix:
        return filename
    return fallback


def _looks_like_telegram_build_request(lower: str) -> bool:
    return any(term in lower for term in ["make a", "build a", "create a", "generate a"]) and any(
        target in lower for target in ["game", "website", "web app", "app", "system", "dashboard", "crm", "management"]
    )


def _starter_file_content(filename: str, message: str) -> str:
    lower = message.lower()
    suffix = Path(filename).suffix.lower()
    if suffix == ".py":
        if "hello" in lower:
            return 'print("Hello, World!")\n'
        return 'print("Created by SHAMSU Telegram")\n'
    if suffix == ".html":
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        return f"<!doctype html>\n<html>\n<head><meta charset=\"utf-8\"><title>{title}</title></head>\n<body><h1>{title}</h1><p>Created by SHAMSU Telegram.</p></body>\n</html>\n"
    if suffix in {".js", ".mjs"}:
        return 'console.log("Created by SHAMSU Telegram");\n'
    if suffix in {".md", ".txt"}:
        return "Created by SHAMSU Telegram.\n"
    return "Created by SHAMSU Telegram.\n"


def _write_workspace_text(filename: str, content: str) -> str:
    rel = filename.replace("\\", "/").lstrip("/")
    if ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail="Telegram file path cannot contain ..")
    suffix = Path(rel).suffix.lower()
    if suffix not in {".py", ".html", ".css", ".js", ".md", ".txt", ".json", ".c", ".h"}:
        raise HTTPException(status_code=400, detail="Telegram can create only safe text/code files")
    path = (settings.workdir_path / rel).resolve()
    root = settings.workdir_path.resolve()
    if root != path and root not in path.parents:
        raise HTTPException(status_code=400, detail="Telegram file path must stay inside workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _fence_language(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".py": "python",
        ".html": "html",
        ".css": "css",
        ".js": "javascript",
        ".json": "json",
        ".c": "c",
        ".h": "c",
        ".md": "markdown",
    }.get(suffix, "")

