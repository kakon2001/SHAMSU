"""Tool implementations for the agent.

Read-only tools (list_directory, read_file, search_files) execute immediately.
Mutating tools (run_shell, write_file) are declared here but the agent loop
gates them behind user approval before calling the executors below.
"""

import asyncio
import difflib
import re
import subprocess
from pathlib import Path

from ..config import settings

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}


def resolve_in_workspace(rel_path: str) -> Path:
    """Resolve a path relative to the workspace root, rejecting escapes."""
    root = settings.workdir_path
    candidate = (root / (rel_path or ".")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path '{rel_path}' is outside the workspace")
    return candidate


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


# ---------------------------------------------------------------------------
# Read-only tools (auto-executed)
# ---------------------------------------------------------------------------

def list_directory(path: str = ".") -> str:
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist"
    if target.is_file():
        return f"'{path}' is a file, not a directory"
    lines = []
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for entry in entries:
        if entry.name in IGNORED_DIRS:
            continue
        rel = entry.relative_to(settings.workdir_path).as_posix()
        lines.append(f"{rel}/" if entry.is_dir() else rel)
    return "\n".join(lines) if lines else "(empty directory)"


def read_file(path: str) -> str:
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist"
    if target.is_dir():
        return f"Error: '{path}' is a directory"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading '{path}': {exc}"
    return _truncate(text, settings.max_read_file_chars)


def search_files(query: str, path: str = ".") -> str:
    """Regex (falling back to literal) search across text files in the workspace."""
    target = resolve_in_workspace(path)
    try:
        pattern = re.compile(query)
    except re.error:
        pattern = re.compile(re.escape(query))

    matches: list[str] = []
    for file in sorted(target.rglob("*")):
        if not file.is_file():
            continue
        if any(part in IGNORED_DIRS for part in file.parts):
            continue
        if file.stat().st_size > 512 * 1024:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        rel = file.relative_to(settings.workdir_path).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(matches) >= 100:
                    matches.append("... [more matches truncated]")
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches found."


# ---------------------------------------------------------------------------
# Mutating tools (executed only after user approval)
# ---------------------------------------------------------------------------

def _run_shell_sync(command: str) -> str:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(settings.workdir_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.shell_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {settings.shell_timeout_seconds}s"

    parts = []
    if proc.stdout and proc.stdout.strip():
        parts.append(proc.stdout.strip())
    if proc.stderr and proc.stderr.strip():
        parts.append(f"[stderr]\n{proc.stderr.strip()}")
    parts.append(f"[exit code: {proc.returncode}]")
    return _truncate("\n".join(parts), settings.max_tool_output_chars)


async def run_shell(command: str) -> str:
    return await asyncio.to_thread(_run_shell_sync, command)


def write_file(path: str, content: str) -> str:
    target = resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to '{path}'."


def make_diff(path: str, new_content: str) -> tuple[str, bool]:
    """Unified diff of the proposed write vs. what's on disk. Returns (diff, is_new_file)."""
    try:
        target = resolve_in_workspace(path)
        old_content = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        is_new = not target.is_file()
    except ValueError:
        raise
    except Exception:
        old_content, is_new = "", True

    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return diff or "(no changes — file content is identical)", is_new


# ---------------------------------------------------------------------------
# Schemas advertised to the model
# ---------------------------------------------------------------------------

# Kept deliberately small and flat: small local models get unreliable when the
# tool surface grows or parameters nest.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories under a path relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to list. Use '.' for the root."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text contents of a file at a path relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of the file to read."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search all text files in the workspace for a regex or literal string. Returns file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or literal text to search for."},
                    "path": {"type": "string", "description": "Relative directory to search in. Defaults to the root."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write the complete new contents of a file (relative path). The user is shown a diff and must "
                "approve it before it is written to disk. Always pass the FULL intended file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of the file to write."},
                    "content": {"type": "string", "description": "The complete new file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a PowerShell command in the workspace root (e.g. run tests, install packages, git, create/"
                "delete/move files). The user must approve the command before it runs. Returns stdout, stderr "
                "and the exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The PowerShell command to execute."}
                },
                "required": ["command"],
            },
        },
    },
]

READ_ONLY_TOOLS = {"list_directory", "read_file", "search_files"}
MUTATING_TOOLS = {"write_file", "run_shell"}
TOOL_NAMES = READ_ONLY_TOOLS | MUTATING_TOOLS
