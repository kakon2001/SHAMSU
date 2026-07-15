"""Lightweight workspace context index.

Text files in the workspace are split into chunks and searched with keyword
overlap. This module also builds compact file, upload, and conversation
summaries so long projects and long sessions can fit into small local model
contexts without sending code to third parties.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}
TEXT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".yaml", ".yml"}
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200
AUTO_CONTEXT_CHARS = 4500
SUMMARY_CONTEXT_CHARS = 2600
CONVERSATION_MEMORY_CHARS = 2200
STOP_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "not",
    "your",
    "file",
    "files",
    "class",
    "function",
    "return",
}


@dataclass
class ContextChunk:
    path: str
    start_line: int
    end_line: int
    text: str
    score: int = 0


def search_context(query: str, limit: int = 5) -> list[ContextChunk]:
    terms = _terms(query)
    if not terms:
        return []
    phrase = query.strip().lower()
    scored: list[ContextChunk] = []
    for chunk in build_workspace_chunks():
        chunk_terms = _terms(chunk.text)
        term_counts = Counter(chunk_terms)
        path_terms = _terms(chunk.path)
        score = sum(term_counts.get(term, 0) for term in terms)
        score += sum(3 for term in terms if term in path_terms)
        if phrase and phrase in chunk.text.lower():
            score += 8
        if score:
            scored.append(ContextChunk(chunk.path, chunk.start_line, chunk.end_line, chunk.text, score))
    scored.sort(key=lambda item: (-item.score, item.path, item.start_line))
    return scored[:limit]


def summarize_workspace(limit: int = 30) -> dict[str, object]:
    files = []
    total_chunks = 0
    total_chars = 0
    upload_count = 0
    for file in sorted(settings.workdir_path.rglob("*")):
        if not _is_indexable_file(file):
            continue
        rel = file.relative_to(settings.workdir_path).as_posix()
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks = _chunk_text(rel, text)
        chunk_count = len(chunks)
        total_chunks += chunk_count
        total_chars += len(text)
        if rel.startswith("uploads/"):
            upload_count += 1
        files.append(
            {
                "path": rel,
                "chars": len(text),
                "chunks": chunk_count,
                "summary": summarize_text(rel, text),
                "detailed_summary": detailed_file_summary(rel, text),
                "top_terms": top_terms(text),
            }
        )
    return {
        "files": files[:limit],
        "file_count": len(files),
        "chunk_count": total_chunks,
        "total_chars": total_chars,
        "uploaded_count": upload_count,
        "summary_budget": SUMMARY_CONTEXT_CHARS,
        "conversation_memory_budget": CONVERSATION_MEMORY_CHARS,
    }


def context_dashboard(limit: int = 12) -> dict[str, object]:
    summary = summarize_workspace(limit=500)
    files = list(summary.get("files", []))
    uploaded = [item for item in files if str(item.get("path", "")).startswith("uploads/")]
    largest = sorted(files, key=lambda item: int(item.get("chars") or 0), reverse=True)[:limit]
    term_counts: Counter[str] = Counter()
    for item in files:
        for term in item.get("top_terms", []) or []:
            term_counts[str(term)] += 1
    return {
        "file_count": summary.get("file_count", 0),
        "chunk_count": summary.get("chunk_count", 0),
        "uploaded_count": len(uploaded),
        "auto_context_budget": AUTO_CONTEXT_CHARS,
        "summary_context_budget": SUMMARY_CONTEXT_CHARS,
        "conversation_memory_budget": CONVERSATION_MEMORY_CHARS,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_terms": [term for term, _ in term_counts.most_common(12)],
        "largest_files": largest,
        "recent_uploads": uploaded[-limit:],
        "file_summaries": files[:limit],
    }


def automatic_context(query: str, limit: int = 6, budget: int = AUTO_CONTEXT_CHARS) -> str:
    matches = search_context(query, limit=limit)
    if not matches:
        return ""
    blocks = []
    used = 0
    for match in matches:
        snippet = match.text.strip()
        header = f"{match.path}:{match.start_line}-{match.end_line}"
        remaining = budget - used - len(header) - 16
        if remaining <= 250:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining] + "\n... [context truncated]"
        block = f"{header}\n{snippet}"
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def automatic_summary_context(query: str, budget: int = SUMMARY_CONTEXT_CHARS) -> str:
    """Return compact summaries of likely relevant files and uploads.

    This complements chunk retrieval: chunks provide exact local evidence, while
    summaries let the model keep a wider project map in a small context window.
    """
    terms = set(_terms(query))
    summary = summarize_workspace(limit=500)
    files = list(summary.get("files", []))
    if not files:
        return ""

    def score(item: dict[str, Any]) -> tuple[int, int, str]:
        path = str(item.get("path") or "")
        text = " ".join([path, str(item.get("summary") or ""), " ".join(item.get("top_terms") or [])])
        item_terms = set(_terms(text))
        relevance = len(terms & item_terms)
        if path.startswith("uploads/"):
            relevance += 4
        return (-relevance, -int(item.get("chars") or 0), path)

    ranked = sorted(files, key=score)
    blocks: list[str] = []
    used = 0
    for item in ranked:
        path = str(item.get("path") or "")
        detail = str(item.get("detailed_summary") or item.get("summary") or "")
        top = ", ".join(str(term) for term in item.get("top_terms", [])[:6])
        block = f"{path} ({item.get('chars', 0)} chars, {item.get('chunks', 0)} chunks)\n{detail}"
        if top:
            block += f"\nKeywords: {top}"
        if used + len(block) + 8 > budget:
            continue
        blocks.append(block)
        used += len(block) + 8
        if used >= budget:
            break
    return "\n\n---\n\n".join(blocks)


def conversation_memory(events: list[dict[str, Any]], current_query: str = "", budget: int = CONVERSATION_MEMORY_CHARS) -> str:
    """Build a compact memory block from earlier prompts, actions, and results."""
    if len(events) < 8:
        return ""
    terms = set(_terms(current_query))
    entries: list[tuple[int, str]] = []
    for event in events:
        kind = str(event.get("type") or "")
        text = _event_summary(event)
        if not text:
            continue
        score = 1
        if kind in {"user_message", "approval_request", "approval_resolved", "files_changed", "error"}:
            score += 2
        score += len(terms & set(_terms(text)))
        entries.append((score, text))

    if not entries:
        return ""
    # Keep high-signal older facts plus the latest events so long sessions remain coherent.
    selected = [text for _, text in sorted(entries, key=lambda item: item[0], reverse=True)[:8]]
    recent = [text for _, text in entries[-8:]]
    merged: list[str] = []
    for text in selected + recent:
        if text not in merged:
            merged.append(text)

    lines: list[str] = []
    used = 0
    for text in merged:
        line = f"- {text}"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def format_context_results(query: str, limit: int = 5) -> str:
    matches = search_context(query, limit=limit)
    if not matches:
        return "No relevant context chunks found."
    blocks = []
    for match in matches:
        snippet = match.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:700] + "\n... [chunk truncated]"
        blocks.append(f"{match.path}:{match.start_line}-{match.end_line} (score {match.score})\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def summarize_text(path: str, text: str, max_chars: int = 260) -> str:
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return "Empty file."
    heading = next((line.lstrip("# ").strip() for line in stripped_lines if line.startswith("#")), "")
    first = heading or stripped_lines[0]
    if len(first) > max_chars:
        first = first[:max_chars].rstrip() + "..."
    kind = "Uploaded context" if path.startswith("uploads/") else "Workspace file"
    return f"{kind}. {first}"


def detailed_file_summary(path: str, text: str, max_chars: int = 520) -> str:
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped:
        return "Empty file."
    facts: list[str] = [summarize_text(path, text, max_chars=180)]
    headings = [line.lstrip("# ").strip() for line in stripped if line.startswith("#")][:4]
    symbols = _code_symbols(text)[:8]
    imports = _import_names(text)[:8]
    terms = top_terms(text, limit=6)
    if headings:
        facts.append("Headings: " + "; ".join(headings))
    if symbols:
        facts.append("Code symbols: " + ", ".join(symbols))
    if imports:
        facts.append("Imports: " + ", ".join(imports))
    if terms:
        facts.append("Main terms: " + ", ".join(terms))
    result = " ".join(facts)
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."
    return result


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(term for term in _terms(text) if term not in STOP_WORDS and len(term) > 2)
    return [term for term, _ in counts.most_common(limit)]


def build_workspace_chunks() -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    for file in sorted(settings.workdir_path.rglob("*")):
        if not _is_indexable_file(file):
            continue
        rel = file.relative_to(settings.workdir_path).as_posix()
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.extend(_chunk_text(rel, text))
    return chunks


def _is_indexable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in IGNORED_DIRS for part in path.parts):
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= 1024 * 1024
    except OSError:
        return False


def _chunk_text(path: str, text: str) -> list[ContextChunk]:
    if not text:
        return []
    line_starts = _line_starts(text)
    chunks: list[ContextChunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        chunks.append(
            ContextChunk(
                path=path,
                start_line=_line_number(line_starts, start),
                end_line=_line_number(line_starts, max(start, end - 1)),
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _event_summary(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind == "user_message":
        return "User asked: " + _short(str(event.get("content") or ""), 180)
    if kind == "assistant_message":
        return "Assistant answered: " + _short(str(event.get("content") or ""), 180)
    if kind == "tool_call":
        return f"Tool used: {event.get('name')} {_short(str(event.get('args') or ''), 140)}"
    if kind == "approval_request":
        target = event.get("path") or event.get("command") or ""
        return f"Approval requested for {event.get('name')}: {_short(str(target), 180)}"
    if kind == "approval_resolved":
        return "Approval accepted" if event.get("approved") else "Approval rejected"
    if kind == "files_changed":
        return "Files changed: " + ", ".join(str(path) for path in event.get("paths") or [])
    if kind == "error":
        return "Error: " + _short(str(event.get("message") or ""), 180)
    return ""


def _code_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in re.finditer(r"^\s*(?:async\s+def|def|class|function|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        name = match.group(1)
        if name not in symbols:
            symbols.append(name)
    return symbols


def _import_names(text: str) -> list[str]:
    imports: list[str] = []
    patterns = [r"^\s*import\s+([A-Za-z0-9_./@-]+)", r"^\s*from\s+([A-Za-z0-9_./@-]+)\s+import"]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            name = match.group(1)
            if name not in imports:
                imports.append(name)
    return imports


def _short(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) > limit:
        return compact[:limit].rstrip() + "..."
    return compact


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def _line_number(starts: list[int], offset: int) -> int:
    line = 1
    for index, start in enumerate(starts, start=1):
        if start > offset:
            break
        line = index
    return line


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[a-zA-Z0-9_]{2,}", text)]
