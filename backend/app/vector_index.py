"""Local semantic vector index for workspace and uploaded context.

This is intentionally dependency-light: text chunks are embedded with a
deterministic hashed bag-of-words vector and stored in SQLite. The interface is
small so the embedder can later be swapped for Ollama or sentence-transformers
without changing the agent, MCP server, or frontend.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import context_index
from .config import settings

VECTOR_DIMS = 256
DEFAULT_LIMIT_FILES = 500
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")
SYNONYMS = {
    "auth": {"login", "logout", "signin", "signup", "token", "password", "session"},
    "login": {"auth", "signin", "account", "session", "token"},
    "database": {"db", "sqlite", "mysql", "postgres", "schema", "table"},
    "error": {"bug", "failure", "exception", "traceback", "fix"},
    "frontend": {"ui", "react", "vite", "browser", "component"},
    "backend": {"api", "fastapi", "server", "route", "endpoint"},
    "game": {"canvas", "animation", "score", "keyboard", "player"},
    "context": {"embedding", "vector", "chunk", "summary", "search"},
}


@dataclass
class VectorMatch:
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    score: float
    preview: str


def rebuild_index(limit_files: int = DEFAULT_LIMIT_FILES) -> dict[str, object]:
    """Rebuild the SQLite vector index from current workspace chunks."""
    limit_files = max(1, min(2000, int(limit_files or DEFAULT_LIMIT_FILES)))
    chunks = context_index.build_workspace_chunks()
    selected: list[context_index.ContextChunk] = []
    seen_files: set[str] = set()
    for chunk in chunks:
        if chunk.path not in seen_files and len(seen_files) >= limit_files:
            continue
        seen_files.add(chunk.path)
        selected.append(chunk)

    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM vector_chunks")
        now = time.time()
        per_file_index: dict[str, int] = {}
        rows = []
        for chunk in selected:
            chunk_index = per_file_index.get(chunk.path, 0)
            per_file_index[chunk.path] = chunk_index + 1
            rows.append(
                (
                    chunk.path,
                    chunk_index,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    json.dumps(embed_text(_embedding_document(chunk))),
                    now,
                )
            )
        conn.executemany(
            """
            INSERT INTO vector_chunks(path, chunk_index, start_line, end_line, text, vector, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return {
        "ok": True,
        "file_count": len(seen_files),
        "chunk_count": len(selected),
        "indexed_files": len(seen_files),
        "indexed_chunks": len(selected),
        "dims": VECTOR_DIMS,
        "db_path": str(db_path),
    }


def stats() -> dict[str, object]:
    db_path = _db_path()
    if not db_path.exists():
        return {"file_count": 0, "chunk_count": 0, "dims": VECTOR_DIMS, "db_path": str(db_path), "ready": False}
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute("SELECT COUNT(DISTINCT path), COUNT(*), MAX(updated_at) FROM vector_chunks").fetchone()
    return {
        "file_count": int(row[0] or 0),
        "chunk_count": int(row[1] or 0),
        "updated_at": float(row[2] or 0.0),
        "dims": VECTOR_DIMS,
        "db_path": str(db_path),
        "ready": bool(row[1]),
    }


def semantic_search(query: str, limit: int = 8) -> list[dict[str, object]]:
    query = query.strip()
    if not query:
        return []
    if int(stats().get("chunk_count") or 0) == 0:
        rebuild_index()

    query_vector = embed_text(query)
    query_terms = set(_tokens(query))
    scored: list[VectorMatch] = []
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT path, chunk_index, start_line, end_line, text, vector FROM vector_chunks"
        ).fetchall()

    for path, chunk_index, start_line, end_line, text, vector_json in rows:
        try:
            vector = json.loads(vector_json)
        except json.JSONDecodeError:
            continue
        score = _dot(query_vector, vector)
        text_terms = set(_tokens(f"{path} {text}"))
        overlap = len(query_terms & text_terms)
        if overlap:
            score += min(0.35, overlap * 0.04)
        if score <= 0:
            continue
        scored.append(
            VectorMatch(
                path=str(path),
                chunk_index=int(chunk_index),
                start_line=int(start_line),
                end_line=int(end_line),
                score=round(float(score), 4),
                preview=_preview(str(text)),
            )
        )

    scored.sort(key=lambda item: (-item.score, item.path, item.start_line))
    return [match.__dict__ for match in scored[: max(1, min(20, int(limit or 8)))]]


def format_semantic_results(query: str, limit: int = 8) -> str:
    matches = semantic_search(query, limit=limit)
    if not matches:
        return "No semantic vector matches found."
    blocks = []
    for match in matches:
        blocks.append(
            f"{match['path']}:{match['start_line']}-{match['end_line']} "
            f"(semantic score {match['score']})\n{match['preview']}"
        )
    return "\n\n---\n\n".join(blocks)


def embed_text(text: str) -> list[float]:
    """Create an offline semantic vector with code-aware weighting.

    It is still deterministic and dependency-light, but stronger than plain
    bag-of-words: identifiers are split, neighboring terms become n-grams,
    and code/project terms receive extra weight.
    """
    vector = [0.0] * VECTOR_DIMS
    for token, weight in _weighted_terms(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % VECTOR_DIMS
        vector[index] += weight
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [round(value / norm, 8) for value in vector]


def _embedding_document(chunk: context_index.ContextChunk) -> str:
    summary = context_index.summarize_text(chunk.path, chunk.text, max_chars=260)
    path_terms = " ".join(_path_terms(chunk.path))
    return f"path {chunk.path}\npath_terms {path_terms}\nsummary {summary}\ncontent\n{chunk.text}"


def _weighted_terms(text: str) -> list[tuple[str, float]]:
    tokens = _expanded_tokens(text)
    weighted: list[tuple[str, float]] = []
    for token in tokens:
        weight = 1.0
        if "_" in token or token in {"class", "function", "def", "api", "route", "schema", "state", "component"}:
            weight += 0.35
        weighted.append((token, weight))
        for part in _identifier_parts(token):
            if part != token:
                weighted.append((part, 0.75))
    for left, right in zip(tokens, tokens[1:]):
        weighted.append((f"{left}_{right}", 0.55))
    for first, second, third in zip(tokens, tokens[1:], tokens[2:]):
        weighted.append((f"{first}_{second}_{third}", 0.3))
    return weighted


def _identifier_parts(token: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).replace("_", "-").replace(".", "-").split("-")
    return [part.lower() for part in parts if len(part) > 1]


def _path_terms(path: str) -> list[str]:
    clean = Path(path)
    parts: list[str] = []
    for piece in clean.parts:
        parts.extend(_identifier_parts(piece))
        parts.extend(_tokens(piece))
    suffix = clean.suffix.lstrip(".").lower()
    if suffix:
        parts.append(suffix)
    return list(dict.fromkeys(parts))

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_db_path())


def _db_path() -> Path:
    history_path = Path(settings.history_db_path)
    if not history_path.is_absolute():
        history_path = settings.workdir_path.parent / history_path
    return history_path.with_name("vector-index.db")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            vector TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_path ON vector_chunks(path)")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _expanded_tokens(text: str) -> list[str]:
    tokens = _tokens(text)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(SYNONYMS.get(token, set()))
    return expanded


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _preview(text: str, limit: int = 520) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "... [chunk truncated]"
