"""
Local Coding Agent - shared core.

Config, tool implementations, model calling, and logging live here so both
the CLI (agent.py) and the web backend (server.py) use identical logic.
"""

import json
import subprocess
import os
import sqlite3
import datetime
from pathlib import Path

import requests

# ---- Config -----------------------------------------------------------

# Ollama exposes an OpenAI-compatible endpoint on port 11434.
# If you switch to mlx_lm.server on a Mac, change this to :8080.
MODEL_SERVER_URL = "http://localhost:11434/v1/chat/completions"

# Override with:
# Windows CMD: set AGENT_MODEL=qwen3:8b
# PowerShell:  $env:AGENT_MODEL="qwen3:8b"
# Mac/Linux:   export AGENT_MODEL=qwen3:8b
MODEL_NAME = os.environ.get("AGENT_MODEL", "qwen3:1.7b")

WORKSPACE_DIR = Path("./workspace").resolve()
LOG_FILE = Path("./activity_log.jsonl")
DB_FILE = Path("./sessions.db").resolve()

WORKSPACE_DIR.mkdir(exist_ok=True)


# ---- Session database -------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_session(session_id: str, state: dict):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO sessions (session_id, state_json, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET "
        "state_json=excluded.state_json, updated_at=excluded.updated_at",
        (session_id, json.dumps(state), datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def load_session(session_id: str) -> dict | None:
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT state_json FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def list_sessions() -> list[dict]:
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT session_id, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"session_id": r[0], "updated_at": r[1]} for r in rows]


init_db()


# ---- System prompt ----------------------------------------------------

SYSTEM_PROMPT = """You are a local coding agent. You have tools to read, write,
edit, and list files, and to run shell commands, all scoped to a workspace
directory.

For NEW files, use write_file. For EDITING an existing file, always prefer
edit_file over write_file - it changes only the part you specify and leaves
the rest of the file untouched, which is safer. When using edit_file, first
use read_file to see the exact current content, since old_text must match
character-for-character, including newlines and indentation.

CRITICAL: Tools that modify files or run commands require user approval.
If the user denies approval, do not claim the action succeeded.

CRITICAL: If a tool result starts with "ERROR", the action did NOT succeed.
You must never tell the user something succeeded when the most recent tool
result was an error. Either fix the problem and retry, or clearly tell the
user what went wrong.

When the task is complete, reply with plain text summarizing what you did -
do not call any more tools."""


# ---- Tool definitions -------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a NEW file, or completely overwrite an existing file, "
                "with given content. This requires user approval before execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a surgical edit to an EXISTING file by replacing one exact "
                "block of text with another, leaving the rest of the file untouched. "
                "old_text must match exactly and must appear exactly once. "
                "This requires user approval before execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {
                        "type": "string",
                        "description": (
                            "The exact existing text to find and replace. "
                            "Must match exactly and appear only once."
                        ),
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The text to replace old_text with.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside the workspace directory. "
                "This requires user approval before execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"],
            },
        },
    },
]


# ---- Permission policy ------------------------------------------------

APPROVAL_REQUIRED_TOOLS = {
    "write_file",
    "edit_file",
    "run_command",
}


def tool_requires_approval(name: str) -> bool:
    return name in APPROVAL_REQUIRED_TOOLS


def describe_tool_call(name: str, args: dict) -> str:
    """Create a human-readable approval preview for risky tool calls."""
    if name == "run_command":
        return f"Run shell command:\n{args.get('command', '')}"

    if name == "write_file":
        content = args.get("content", "")
        preview = content[:500]
        if len(content) > 500:
            preview += "\n... [content truncated]"

        return (
            "Create or overwrite file:\n"
            f"Path: {args.get('path', '')}\n\n"
            f"Content preview:\n{preview}"
        )

    if name == "edit_file":
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        old_preview = old_text[:400]
        new_preview = new_text[:400]

        if len(old_text) > 400:
            old_preview += "\n... [old text truncated]"
        if len(new_text) > 400:
            new_preview += "\n... [new text truncated]"

        return (
            "Edit file:\n"
            f"Path: {args.get('path', '')}\n\n"
            f"Replace:\n{old_preview}\n\n"
            f"With:\n{new_preview}"
        )

    return f"Tool: {name}\nArgs: {json.dumps(args, indent=2)}"


# ---- Tool implementations --------------------------------------------

def _safe_path(rel_path: str) -> Path:
    """Resolve a path and make sure it stays inside WORKSPACE_DIR."""
    p = (WORKSPACE_DIR / rel_path).resolve()

    try:
        p.relative_to(WORKSPACE_DIR)
    except ValueError as exc:
        raise PermissionError(f"Path escapes workspace: {rel_path}") from exc

    return p


def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    p = _safe_path(path)

    if not p.exists():
        return f"ERROR: file not found: {path}. Use write_file to create a new file."
    if not p.is_file():
        return f"ERROR: not a file: {path}"

    current = p.read_text(encoding="utf-8")
    count = current.count(old_text)

    # Fallback: some smaller models emit literal backslash-n instead of real
    # newlines in tool call JSON. If the raw match fails, try converting them.
    if count == 0:
        normalized_old = old_text.replace("\\n", "\n").replace("\\t", "\t")
        normalized_count = current.count(normalized_old)

        if normalized_count >= 1:
            old_text = normalized_old
            new_text = new_text.replace("\\n", "\n").replace("\\t", "\t")
            count = normalized_count

    if count == 0:
        return (
            f"ERROR: old_text not found in {path}. It must match the file's "
            "current content EXACTLY, including whitespace and line breaks. "
            "Use read_file to see the exact current content first."
        )

    if count > 1:
        return (
            f"ERROR: old_text appears {count} times in {path}, but it must be "
            "unique. Include more surrounding context in old_text so it matches "
            "only one location."
        )

    updated = current.replace(old_text, new_text, 1)
    p.write_text(updated, encoding="utf-8")

    return f"OK: replaced {len(old_text)} chars with {len(new_text)} chars in {path}"


def list_dir(path: str = ".") -> str:
    p = _safe_path(path)

    if not p.exists():
        return f"ERROR: no such directory: {path}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"

    entries = sorted(os.listdir(p))
    return "\n".join(entries) if entries else "(empty)"


def execute_command(command: str) -> str:
    """Run a shell command with NO approval check. Caller must gate this."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s"


def run_tool(name: str, args: dict) -> str:
    """
    Execute a safe/approved tool.

    Approval must be handled before this function is called for risky tools.
    run_command is intentionally excluded and must call execute_command().
    """
    try:
        if name == "read_file":
            return read_file(args["path"])

        if name == "write_file":
            return write_file(args["path"], args["content"])

        if name == "edit_file":
            return edit_file(args["path"], args["old_text"], args["new_text"])

        if name == "list_dir":
            return list_dir(args.get("path", "."))

        if name == "run_command":
            raise RuntimeError("run_command must be handled by execute_command() after approval.")

        return f"ERROR: unknown tool {name}"

    except Exception as exc:
        return f"ERROR: tool '{name}' failed: {exc}"


# ---- Logging ----------------------------------------------------------

def log_event(event: dict):
    event["_ts"] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---- Context management -----------------------------------------------

def estimate_tokens(messages: list) -> int:
    """Estimate token count based on character count. Roughly 4 chars/token."""
    total_chars = 0

    for msg in messages:
        if msg.get("content"):
            total_chars += len(msg["content"])
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))

    return total_chars // 4


def group_messages_into_turns(messages: list) -> list:
    """
    Group history messages into atomic turns.

    A turn starts with a user/system message and includes following assistant/tool
    messages until the next user/system message.
    """
    turns = []
    current_turn = []

    for msg in messages:
        role = msg.get("role")

        if role in ("user", "system"):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    return turns


def call_model_raw(messages: list) -> str:
    """Make a simple text completion request to Ollama."""
    resp = requests.post(
        MODEL_SERVER_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
            "think": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def manage_context(messages: list, max_tokens: int = 6000) -> list:
    """
    Condense the message list if it exceeds the maximum token budget.

    Preserves the initial system prompt and the most recent turns. Summarizes
    older turns using the model itself to keep the conversation running longer.
    """
    if len(messages) <= 2:
        return messages

    current_tokens = estimate_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    system_prompt = messages[0]
    history = messages[1:]

    turns = group_messages_into_turns(history)
    if not turns:
        return messages

    keep_turns_count = 3

    if len(turns) <= keep_turns_count:
        if len(turns) > 1:
            keep_turns_count = 1
        else:
            return messages

    old_turns = turns[:-keep_turns_count]
    new_turns = turns[-keep_turns_count:]

    old_messages = []
    for turn in old_turns:
        old_messages.extend(turn)

    summary_content = None

    try:
        summary_prompt_messages = [
            system_prompt,
            *old_messages,
            {
                "role": "user",
                "content": (
                    "Summarize the conversation history above in 2-3 sentences. "
                    "List which files were created/edited, which commands were executed, "
                    "and the current state of the task. Do not refer to system rules or tools."
                ),
            },
        ]
        summary_content = call_model_raw(summary_prompt_messages)
    except Exception:
        pass

    managed_messages = [system_prompt]

    if summary_content:
        managed_messages.append({
            "role": "system",
            "content": f"Summary of previous progress:\n{summary_content}",
        })
    else:
        managed_messages.append({
            "role": "system",
            "content": "[Older conversation history truncated to preserve local context window]",
        })

    for turn in new_turns:
        managed_messages.extend(turn)

    return managed_messages


# ---- Model call -------------------------------------------------------

def call_model(messages: list) -> dict:
    original_tokens = estimate_tokens(messages)
    managed_messages = manage_context(messages)
    new_tokens = estimate_tokens(managed_messages)

    if new_tokens < original_tokens:
        log_event({
            "type": "context_condensed",
            "before_tokens": original_tokens,
            "after_tokens": new_tokens,
        })

        print(
            f"\n[CONTEXT HANDLER] Condensed history from "
            f"{original_tokens} to {new_tokens} estimated tokens."
        )

        messages[:] = managed_messages

    resp = requests.post(
        MODEL_SERVER_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.2,
            "think": False,
            "max_tokens": 300,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def parse_tool_call(call: dict):
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    return name, args
