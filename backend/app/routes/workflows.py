from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..agent import tools
from ..config import settings

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

MAX_RELEVANT_FILES = 8
MAX_RANGE_WINDOWS = 10
RANGE_RADIUS = 35
LARGE_FILE_LINES = 1000
LARGE_FILE_BYTES = 300_000


class EditWorkflowRequest(BaseModel):
    prompt: str
    path: str = "."
    query: str | None = None


class RelevantFile(BaseModel):
    path: str
    language: str
    lines: int | None = None
    bytes: int
    indexed: bool
    large: bool
    reason: str
    symbols: list[str] = []
    imports: list[str] = []


class RangeWindow(BaseModel):
    path: str
    start_line: int
    end_line: int
    reason: str
    preview: str


class PatchPlanItem(BaseModel):
    path: str
    strategy: str
    reason: str
    tool: str


class EditWorkflowResponse(BaseModel):
    goal: str
    index_summary: str
    relevant_files: list[RelevantFile]
    range_windows: list[RangeWindow]
    patch_plan: list[PatchPlanItem]
    impact_summary: list[str]
    verification_commands: list[str]
    next_steps: list[str]


@router.post("/edit-plan", response_model=EditWorkflowResponse)
def plan_large_edit(body: EditWorkflowRequest) -> EditWorkflowResponse:
    """Plan a large-file or multi-file edit without mutating the workspace."""
    index = _load_project_index(body.path)
    query = (body.query or _query_from_prompt(body.prompt)).strip()
    relevant = _select_relevant_files(index, body.prompt, query)
    windows = _range_windows(relevant, query)
    patch_plan = _patch_plan(relevant, body.prompt)
    verification = _verification_commands(relevant)
    return EditWorkflowResponse(
        goal=body.prompt.strip(),
        index_summary=_index_summary(index),
        relevant_files=relevant,
        range_windows=windows,
        patch_plan=patch_plan,
        impact_summary=_impact_summary(relevant, body.prompt),
        verification_commands=verification,
        next_steps=[
            "Review the selected line ranges before editing.",
            "Use replace_in_file for exact-block patches instead of rewriting large files.",
            "Run the suggested verification commands after each focused edit.",
            "If verification fails, feed the error back into this workflow and patch the smallest affected block.",
        ],
    )


def _load_project_index(path: str) -> dict[str, Any]:
    raw = tools.project_index(path)
    if raw.startswith("Error:"):
        return {"root": path, "file_count": 0, "files": [], "error": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"root": path, "file_count": 0, "files": [], "error": "project_index returned invalid JSON"}
    if not isinstance(parsed, dict):
        return {"root": path, "file_count": 0, "files": [], "error": "project_index returned a non-object payload"}
    return parsed


def _query_from_prompt(prompt: str) -> str:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", prompt)
    ignored = {"the", "and", "for", "with", "file", "files", "project", "change", "update", "implement", "large", "huge", "fix", "bug", "error"}
    useful = [word for word in words if word.lower() not in ignored]
    return "|".join(useful[:6]) or prompt.strip()[:80]


def _select_relevant_files(index: dict[str, Any], prompt: str, query: str) -> list[RelevantFile]:
    files = [item for item in index.get("files", []) if isinstance(item, dict)]
    scored: list[tuple[int, dict[str, Any], str]] = []
    prompt_lower = prompt.lower()
    query_terms = [term.lower() for term in re.split(r"\W+", query) if term]
    search_hits = _search_hit_paths(query) if query else {}

    for item in files:
        path = str(item.get("path") or "")
        if not path:
            continue
        symbols = [str(value) for value in item.get("symbols") or []]
        imports = [str(value) for value in item.get("imports") or []]
        haystack = " ".join([path, *symbols, *imports]).lower()
        score = 0
        reasons: list[str] = []
        if path.lower() in prompt_lower:
            score += 8
            reasons.append("path is mentioned in the prompt")
        for term in query_terms:
            if term and term in haystack:
                score += 3
        if path in search_hits:
            score += 6 + min(search_hits[path], 4)
            reasons.append("query appears in file contents")
        if any(keyword in prompt_lower for keyword in ["frontend", "ui", "react"]) and Path(path).suffix.lower() in {".tsx", ".jsx", ".css", ".html"}:
            score += 2
            reasons.append("frontend-related file type")
        if any(keyword in prompt_lower for keyword in ["backend", "api", "route", "database"]) and Path(path).suffix.lower() == ".py":
            score += 2
            reasons.append("backend-related file type")
        if score == 0 and len(scored) < 3:
            score = 1
            reasons.append("included from project index for orientation")
        if score > 0:
            scored.append((score, item, "; ".join(reasons) or "matches prompt/query terms"))

    selected = sorted(scored, key=lambda row: (-row[0], str(row[1].get("path") or "")))[:MAX_RELEVANT_FILES]
    return [_file_model(item, reason) for _, item, reason in selected]


def _search_hit_paths(query: str) -> dict[str, int]:
    raw = tools.search_files(query or ".")
    hits: dict[str, int] = {}
    if raw.startswith("No matches") or raw.startswith("Error:"):
        return hits
    for line in raw.splitlines():
        match = re.match(r"([^:]+):(\d+):", line)
        if match:
            hits[match.group(1)] = hits.get(match.group(1), 0) + 1
    return hits


def _file_model(item: dict[str, Any], reason: str) -> RelevantFile:
    path = str(item.get("path") or "")
    suffix = Path(path).suffix.lower().lstrip(".") or "text"
    lines = item.get("lines") if isinstance(item.get("lines"), int) else None
    size = int(item.get("bytes") or 0)
    large = bool(size >= LARGE_FILE_BYTES or (lines is not None and lines >= LARGE_FILE_LINES) or not item.get("indexed", False))
    return RelevantFile(
        path=path,
        language=suffix,
        lines=lines,
        bytes=size,
        indexed=bool(item.get("indexed", False)),
        large=large,
        reason=reason,
        symbols=[str(value) for value in item.get("symbols") or []][:10],
        imports=[str(value) for value in item.get("imports") or []][:10],
    )


def _range_windows(files: list[RelevantFile], query: str) -> list[RangeWindow]:
    windows: list[RangeWindow] = []
    terms = [term for term in re.split(r"\W+", query) if term]
    for file in files:
        if len(windows) >= MAX_RANGE_WINDOWS:
            break
        hit_lines = _line_hits(file.path, terms)
        if hit_lines:
            for line in hit_lines[:2]:
                if len(windows) >= MAX_RANGE_WINDOWS:
                    break
                start = max(1, line - RANGE_RADIUS)
                end = line + RANGE_RADIUS
                windows.append(RangeWindow(path=file.path, start_line=start, end_line=end, reason=f"around match at line {line}", preview=tools.read_file_range(file.path, start, end)))
        else:
            end = min(file.lines or 120, 120)
            windows.append(RangeWindow(path=file.path, start_line=1, end_line=end, reason="top of relevant file for orientation", preview=tools.read_file_range(file.path, 1, end)))
    return windows


def _line_hits(path: str, terms: list[str]) -> list[int]:
    if not terms:
        return []
    target = tools.resolve_in_workspace(path)
    if not target.exists() or not target.is_file() or target.stat().st_size > 1024 * 1024:
        return []
    pattern = re.compile("|".join(re.escape(term) for term in terms), flags=re.IGNORECASE)
    hits: list[int] = []
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                if pattern.search(line):
                    hits.append(lineno)
                    if len(hits) >= 5:
                        break
    except OSError:
        return []
    return hits


def _patch_plan(files: list[RelevantFile], prompt: str) -> list[PatchPlanItem]:
    prompt_lower = prompt.lower()
    plan: list[PatchPlanItem] = []
    for file in files:
        if file.large:
            strategy = "Patch the smallest exact text block found from read_file_range output; do not rewrite the whole file."
        elif any(word in prompt_lower for word in ["add", "implement", "create"]):
            strategy = "Use replace_in_file for the target function/class, or write_file only for a new small companion file."
        else:
            strategy = "Use replace_in_file after confirming the exact old_text from the selected range."
        plan.append(PatchPlanItem(path=file.path, strategy=strategy, reason=file.reason, tool="replace_in_file"))
    return plan


def _impact_summary(files: list[RelevantFile], prompt: str) -> list[str]:
    if not files:
        return ["No relevant files found. Run project_index/search_files with a more specific query before editing."]
    summary = [f"{file.path}: {file.reason}." for file in files[:5]]
    if len(files) > 1:
        summary.append("Because multiple files are involved, edit and verify one small behavior at a time.")
    if any(file.large for file in files):
        summary.append("At least one selected file is large or unindexed, so use bounded range reads and exact-block patches only.")
    return summary


def _verification_commands(files: list[RelevantFile]) -> list[str]:
    suffixes = {Path(file.path).suffix.lower() for file in files}
    commands: list[str] = []
    if ".py" in suffixes:
        commands.append("python -m py_compile <changed_file.py>")
        commands.append("python -m pytest -q")
    if suffixes & {".ts", ".tsx", ".js", ".jsx", ".html", ".css"}:
        commands.append("npm run build")
    if ".c" in suffixes:
        commands.append("gcc <changed_file.c> -o <output.exe>")
    if not commands:
        commands.append("Re-open the changed file and confirm the expected text is present.")
    return commands


def _index_summary(index: dict[str, Any]) -> str:
    if index.get("error"):
        return str(index["error"])
    files = [item for item in index.get("files", []) if isinstance(item, dict)]
    large = sum(1 for item in files if not item.get("indexed", False) or int(item.get("bytes") or 0) >= LARGE_FILE_BYTES)
    return f"Indexed {index.get('file_count', len(files))} files; {large} large/unindexed files need bounded range reads."
