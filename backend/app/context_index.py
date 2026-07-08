"""Lightweight workspace context index.

Text files in the workspace are split into chunks and searched with keyword
overlap. This is a simple local foundation for context engineering; a future
version can replace the scoring with embeddings or a vector database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}
TEXT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".yaml", ".yml"}
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200


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
    scored: list[ContextChunk] = []
    for chunk in build_workspace_chunks():
        haystack = " ".join(_terms(chunk.path + "\n" + chunk.text))
        score = sum(haystack.count(term) for term in terms)
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
        chunk_count = len(_chunk_text(rel, text))
        total_chunks += chunk_count
        files.append({"path": rel, "chars": len(text), "chunks": chunk_count})
    return {"files": files[:limit], "file_count": len(files), "chunk_count": total_chunks}


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
