"""High-signal long-context bundle for large SHAMSU project questions.

The regular context index can return exact keyword chunks, and the vector index can
return semantic matches. This module combines them with project memory and symbol
hints so the agent gets a compact Claude-like orientation packet before working
on broad or vague project requests.
"""

from __future__ import annotations

import re
from typing import Any

from . import context_index, vector_index

DEFAULT_BUDGET = 7000
SYMBOL_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+def|def|class|function|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def build_bundle(query: str, budget: int = DEFAULT_BUDGET, fast: bool = False) -> dict[str, object]:
    query = (query or "").strip()
    budget = max(2500, min(16000, int(budget or DEFAULT_BUDGET)))
    project = _fast_project_snapshot(query) if fast else context_index.project_map(limit=120)
    keyword_matches = _fast_keyword_matches(query, limit=5) if fast and query else (context_index.search_context(query, limit=8) if query else [])
    semantic_matches = [] if fast else (_safe_semantic_matches(query, limit=8) if query else [])
    symbols = _fast_symbol_matches(query, limit=8) if fast and query else (symbol_matches(query, limit=12) if query else [])
    memory = project_memory(project, query, budget=1400 if fast else 1800)

    return {
        "query": query,
        "budget": budget,
        "memory": memory,
        "symbols": symbols,
        "keyword_matches": [
            {
                "path": match.path,
                "start_line": match.start_line,
                "end_line": match.end_line,
                "score": match.score,
                "preview": _compact(match.text, 520),
            }
            for match in keyword_matches
        ],
        "semantic_matches": semantic_matches,
        "project": {
            "languages": project.get("languages") or {},
            "frameworks": project.get("frameworks") or [],
            "entry_points": project.get("entry_points") or [],
            "important_files": project.get("important_files") or [],
            "large_files": project.get("large_files") or [],
        },
        "context": format_bundle_parts(memory, symbols, keyword_matches, semantic_matches, project, budget=budget),
    }


def format_bundle(query: str, budget: int = DEFAULT_BUDGET) -> str:
    return str(build_bundle(query, budget=budget).get("context") or "")




def _fast_project_snapshot(query: str = "") -> dict[str, object]:
    terms = set(context_index._terms(query))
    important: list[dict[str, object]] = []
    for rel, text in _iter_fast_text_files(max_files=120):
        haystack_terms = set(context_index._terms(rel + " " + text[:1200]))
        score = len(terms & haystack_terms)
        if score or len(important) < 10:
            important.append(
                {
                    "path": rel,
                    "language": "text",
                    "role": "context",
                    "lines": text.count("\n") + (1 if text else 0),
                    "summary": context_index.summarize_text(rel, text),
                    "score": score,
                }
            )
    important.sort(key=lambda item: (-int(item.get("score") or 0), str(item.get("path") or "")))
    return {"languages": {}, "frameworks": [], "entry_points": [], "important_files": important[:12], "large_files": []}


def _fast_keyword_matches(query: str, limit: int = 5) -> list[context_index.ContextChunk]:
    terms = set(context_index._terms(query))
    matches: list[context_index.ContextChunk] = []
    for rel, text in _iter_fast_text_files(max_files=160):
        haystack = (rel + "\n" + text).lower()
        score = sum(1 for term in terms if term in haystack)
        if not score:
            continue
        preview = text[:1600]
        matches.append(context_index.ContextChunk(rel, 1, min(80, text.count("\n") + 1), preview, score))
    matches.sort(key=lambda item: (-item.score, item.path))
    return matches[:limit]


def _fast_symbol_matches(query: str, limit: int = 8) -> list[dict[str, object]]:
    terms = set(context_index._terms(query))
    matches: list[dict[str, object]] = []
    for rel, text in _iter_fast_text_files(max_files=160):
        path_terms = set(context_index._terms(rel))
        for symbol in SYMBOL_RE.findall(text[:20000]):
            symbol_terms = set(context_index._terms(_split_identifier(symbol)))
            score = len(terms & symbol_terms) * 5 + len(terms & path_terms) * 2
            if score:
                matches.append({"path": rel, "symbol": symbol, "start_line": 1, "end_line": 200, "score": score})
    matches.sort(key=lambda item: (-int(item["score"]), str(item["path"]), str(item["symbol"])))
    return matches[:limit]


def _iter_fast_text_files(max_files: int = 160):
    count = 0
    for file in sorted(context_index.settings.workdir_path.rglob("*")):
        if count >= max_files:
            break
        if not file.is_file() or any(part in context_index.IGNORED_DIRS for part in file.parts):
            continue
        if file.suffix.lower() not in context_index.TEXT_EXTENSIONS:
            continue
        try:
            if file.stat().st_size > 256 * 1024:
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        count += 1
        yield file.relative_to(context_index.settings.workdir_path).as_posix(), text
def project_memory(project: dict[str, object], query: str = "", budget: int = 1800) -> str:
    lines: list[str] = []
    languages = project.get("languages") or {}
    if languages:
        lines.append("Languages: " + ", ".join(f"{name}={count}" for name, count in dict(languages).items()))
    frameworks = project.get("frameworks") or []
    if frameworks:
        lines.append("Framework hints: " + ", ".join(str(item) for item in frameworks[:8]))
    entry_points = project.get("entry_points") or []
    if entry_points:
        lines.append("Entry points: " + ", ".join(str(item.get("path")) for item in entry_points[:8] if isinstance(item, dict)))
    important = project.get("important_files") or []
    if important:
        lines.append("Important files:")
        for item in important[:10]:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or "")
            lines.append(f"- {item.get('path')} [{item.get('language')}, {item.get('role')}, {item.get('lines')} lines] {summary}")
    large = project.get("large_files") or []
    if large:
        lines.append("Large files needing range reads: " + ", ".join(str(item.get("path")) for item in large[:8] if isinstance(item, dict)))
    result = "\n".join(lines)
    if len(result) > budget:
        result = result[:budget].rstrip() + "\n... [project memory truncated]"
    return result


def _safe_semantic_matches(query: str, limit: int = 8) -> list[dict[str, object]]:
    try:
        if not vector_index.stats().get("ready"):
            return []
        return vector_index.semantic_search(query, limit=limit)
    except Exception:
        return []


def symbol_matches(query: str, limit: int = 12) -> list[dict[str, object]]:
    terms = set(context_index._terms(query))  # local tokenizer used by the existing context index
    if not terms:
        return []
    matches: list[dict[str, object]] = []
    for chunk in context_index.build_workspace_chunks():
        path_terms = set(context_index._terms(chunk.path))
        for symbol in SYMBOL_RE.findall(chunk.text):
            symbol_terms = set(context_index._terms(_split_identifier(symbol)))
            score = len(terms & symbol_terms) * 5 + len(terms & path_terms) * 2
            if score:
                matches.append(
                    {
                        "path": chunk.path,
                        "symbol": symbol,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "score": score,
                    }
                )
    matches.sort(key=lambda item: (-int(item["score"]), str(item["path"]), str(item["symbol"])))
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in matches:
        key = (str(item["path"]), str(item["symbol"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def format_bundle_parts(
    memory: str,
    symbols: list[dict[str, object]],
    keyword_matches: list[context_index.ContextChunk],
    semantic_matches: list[dict[str, object]],
    project: dict[str, object],
    budget: int = DEFAULT_BUDGET,
) -> str:
    sections: list[str] = []
    if memory:
        sections.append("Project memory\n" + memory)
    if symbols:
        lines = ["Symbol matches"]
        for item in symbols[:10]:
            lines.append(f"- {item['path']}:{item['start_line']}-{item['end_line']} symbol {item['symbol']} score {item['score']}")
        sections.append("\n".join(lines))
    if semantic_matches:
        lines = ["Semantic matches"]
        for item in semantic_matches[:6]:
            lines.append(f"- {item.get('path')}:{item.get('start_line')}-{item.get('end_line')} score {item.get('score')}\n  {_compact(str(item.get('preview') or ''), 360)}")
        sections.append("\n".join(lines))
    if keyword_matches:
        lines = ["Exact keyword chunks"]
        for item in keyword_matches[:6]:
            lines.append(f"- {item.path}:{item.start_line}-{item.end_line} score {item.score}\n  {_compact(item.text, 360)}")
        sections.append("\n".join(lines))
    result = "\n\n---\n\n".join(sections)
    if len(result) > budget:
        result = result[:budget].rstrip() + "\n... [long-context bundle truncated]"
    return result or "No long-context matches found."


def _compact(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _split_identifier(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    return spaced.replace("_", " ").replace("-", " ")