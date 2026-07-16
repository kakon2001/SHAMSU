"""Tool implementations for the agent.

Read-only tools (list_directory, read_file, search_files) execute immediately.
Mutating tools (run_shell, write_file) are declared here but the agent loop
gates them behind user approval before calling the executors below.
"""

import asyncio
import difflib
import os
import platform
import re
import subprocess
from pathlib import Path

from .. import context_index
from ..config import settings

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}

BLOCKED_COMMAND_PATTERNS = [
    r"\bRemove-Item\b.*\s-(?:Recurse|r)\b.*\s-(?:Force|f)\b",
    r"\brm\b.*\s-rf\b",
    r"\bdel\b.*\s/[sq]\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\brestart-computer\b",
    r"\bstop-computer\b",
    r"\breg\s+(?:add|delete|import)\b",
    r"\bSet-ExecutionPolicy\b",
]

RISKY_COMMAND_PATTERNS = [
    r"\bRemove-Item\b",
    r"\brm\b",
    r"\bdel\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+clean\b",
    r"\bgit\s+push\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+install\b",
    r"\bInvoke-WebRequest\b",
    r"\bcurl(?:\.exe)?\b",
]


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


def analyze_shell_command(command: str) -> dict[str, str | bool]:
    normalized = command.strip()
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return {
                "allowed": False,
                "risk": "blocked",
                "reason": "Command matches a blocked destructive/system-level pattern.",
            }
    for pattern in RISKY_COMMAND_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return {
                "allowed": True,
                "risk": "high",
                "reason": "Command may change files, install packages, access the network, or alter Git state.",
            }
    return {"allowed": True, "risk": "normal", "reason": "No high-risk pattern detected."}


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


def search_context(query: str) -> str:
    return context_index.format_context_results(query)




def read_file_range(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read a bounded line range from a large workspace file."""
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if end_line - start_line > 500:
        end_line = start_line + 500
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist"
    if target.is_dir():
        return f"Error: '{path}' is a directory"
    lines: list[str] = []
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                if lineno < start_line:
                    continue
                if lineno > end_line:
                    break
                lines.append(f"{lineno}: {line.rstrip()}")
    except Exception as exc:
        return f"Error reading '{path}': {exc}"
    return "\n".join(lines) if lines else f"No lines found in requested range {start_line}-{end_line}."


def project_index(path: str = ".") -> str:
    """Return a compact local project index with files, sizes, symbols, and imports."""
    root = resolve_in_workspace(path)
    if not root.exists():
        return f"Error: '{path}' does not exist"
    files: list[dict[str, object]] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or any(part in IGNORED_DIRS for part in file.parts):
            continue
        rel = file.relative_to(settings.workdir_path).as_posix()
        try:
            size = file.stat().st_size
        except OSError:
            continue
        if size > 1024 * 1024 or file.suffix.lower() not in {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".md", ".txt", ".json"}:
            files.append({"path": rel, "bytes": size, "indexed": False})
            continue
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        files.append(
            {
                "path": rel,
                "bytes": size,
                "lines": content.count("\n") + (1 if content else 0),
                "indexed": True,
                "symbols": _code_symbols(content)[:20],
                "imports": _import_names(content)[:20],
            }
        )
    return json_dumps({"root": path, "file_count": len(files), "files": files[:300]})


def make_patch_diff(path: str, old_text: str, new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    ) or "(no changes)"


def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Patch-style edit: replace one exact text block in a file."""
    target = resolve_in_workspace(path)
    if not target.exists() or not target.is_file():
        return f"Error: '{path}' is not a file"
    current = target.read_text(encoding="utf-8", errors="replace")
    if old_text not in current:
        return "Error: old_text was not found exactly; use read_file_range/search_files to locate the current text."
    updated = current.replace(old_text, new_text, 1)
    target.write_text(updated, encoding="utf-8")
    return f"Patched '{path}' by replacing {len(old_text)} characters with {len(new_text)} characters."


def make_replace_diff(path: str, old_text: str, new_text: str) -> tuple[str, bool]:
    target = resolve_in_workspace(path)
    current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    if old_text not in current:
        return "Error: old_text was not found exactly; no patch can be previewed.", False
    updated = current.replace(old_text, new_text, 1)
    return make_patch_diff(path, current, updated), False


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _code_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    patterns = [
        r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*(?:export\s+)?(?:function|const|let|var|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            name = match.group(1)
            if name not in symbols:
                symbols.append(name)
    return symbols


def _import_names(text: str) -> list[str]:
    imports: list[str] = []
    patterns = [
        r"^\s*import\s+([A-Za-z0-9_./@-]+)",
        r"^\s*from\s+([A-Za-z0-9_./@-]+)\s+import",
        r"from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            name = match.group(1)
            if name not in imports:
                imports.append(name)
    return imports

# ---------------------------------------------------------------------------
# Mutating tools (executed only after user approval)
# ---------------------------------------------------------------------------

def _run_shell_sync(command: str) -> str:
    shell_command = _shell_invocation(command)
    try:
        proc = subprocess.run(
            shell_command,
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


def _shell_invocation(command: str) -> list[str]:
    if platform.system().lower().startswith("win"):
        return ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]
    shell = os.environ.get("SHELL")
    if not shell:
        shell = "/bin/zsh" if Path("/bin/zsh").exists() else "/bin/bash"
    return [shell, "-lc", command]


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
    return diff or "(no changes â€” file content is identical)", is_new


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
            "name": "search_context",
            "description": (
                "Search chunked workspace context for relevant snippets. Use this when the user asks broad "
                "questions about the project or needs context across multiple files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or keyword query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "Read a bounded line range from a large file. Use this for huge files instead of read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_index",
            "description": "Build a compact index of project files, sizes, symbols, and imports for multi-file reasoning.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative directory, default root."}},
                "required": [],
            },
        },
    },    {
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
            "name": "replace_in_file",
            "description": "Patch-based edit: replace one exact text block in a file. User sees a diff and must approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the workspace root (e.g. run tests, install packages, git, create/"
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

READ_ONLY_TOOLS = {"list_directory", "read_file", "read_file_range", "search_files", "search_context", "project_index"}
MUTATING_TOOLS = {"write_file", "replace_in_file", "run_shell"}
TOOL_NAMES = READ_ONLY_TOOLS | MUTATING_TOOLS

