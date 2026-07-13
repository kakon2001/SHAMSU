"""Lightweight workspace context index.

Text files in the workspace are split into chunks and searched with keyword
overlap. This is a simple local foundation for context engineering; a future
version can replace the scoring with embeddings or a vector database.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import settings

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}
TEXT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".yaml", ".yml"}
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200
AUTO_CONTEXT_CHARS = 4500
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
        files.append(
            {
                "path": rel,
                "chars": len(text),
                "chunks": chunk_count,
                "summary": summarize_text(rel, text),
                "top_terms": top_terms(text),
            }
        )
    return {"files": files[:limit], "file_count": len(files), "chunk_count": total_chunks}



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
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_terms": [term for term, _ in term_counts.most_common(12)],
        "largest_files": largest,
        "recent_uploads": uploaded[-limit:],
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

