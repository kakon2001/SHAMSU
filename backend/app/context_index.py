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
    """Build a smooth, categorized memory block from previous session events.

    Instead of replaying the transcript, keep the facts that help the next turn:
    earlier goals, files touched, approvals/results, blockers, and the latest few
    actions. This gives small local models Claude-like continuity without flooding
    the context window.
    """
    snapshot = conversation_memory_snapshot(events, current_query, budget=budget)
    sections: list[str] = []
    emitted: set[str] = set()
    relevant = _section_unique(snapshot["relevant"], emitted)
    files_touched = _section_unique(snapshot["files_touched"], emitted)
    recent = _section_unique(snapshot["recent"], emitted)
    if relevant:
        sections.append("Relevant earlier context:\n" + "\n".join(f"- {item}" for item in relevant))
    if files_touched:
        sections.append("Files touched this session:\n" + "\n".join(f"- {item}" for item in files_touched))
    if recent:
        sections.append("Recent continuity:\n" + "\n".join(f"- {item}" for item in recent))
    result = "\n\n".join(sections)
    if len(result) > budget:
        result = result[:budget].rstrip() + "\n... [conversation memory truncated]"
    return result


def conversation_memory_snapshot(events: list[dict[str, Any]], current_query: str = "", budget: int = CONVERSATION_MEMORY_CHARS) -> dict[str, list[str]]:
    if len(events) < 4:
        return {"relevant": [], "recent": [], "files_touched": []}
    terms = set(_terms(current_query))
    scored: list[tuple[int, int, str]] = []
    recent_candidates: list[str] = []
    touched: list[str] = []
    for index, event in enumerate(events):
        text = _event_summary(event)
        if not text:
            continue
        kind = str(event.get("type") or "")
        score = _memory_event_weight(kind) + len(terms & set(_terms(text))) * 4 + min(index, 40) // 10
        scored.append((score, index, text))
        if kind in {"user_message", "assistant_message", "tool_result", "error", "files_changed"}:
            recent_candidates.append(text)
        if kind == "files_changed":
            for path in event.get("paths") or []:
                path_text = str(path)
                if path_text and path_text not in touched:
                    touched.append(path_text)
        elif kind == "approval_request" and event.get("path"):
            path_text = str(event.get("path"))
            if path_text not in touched:
                touched.append(path_text)

    recent = _dedupe_memory_items(recent_candidates[-8:])[-6:]
    recent_keys = {" ".join(item.split()).lower() for item in recent}
    relevant_pool = _dedupe_memory_items([text for _, _, text in sorted(scored, key=lambda item: (-item[0], -item[1]))])
    relevant = [item for item in relevant_pool if " ".join(item.split()).lower() not in recent_keys][:8]
    if not relevant:
        relevant = relevant_pool[: min(2, len(relevant_pool))]
    files_touched = touched[-10:]
    return _fit_memory_snapshot({"relevant": relevant, "recent": recent, "files_touched": files_touched}, budget)



def _section_unique(items: list[str], emitted: set[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        key = " ".join(item.split()).lower()
        if key in emitted:
            continue
        emitted.add(key)
        result.append(item)
    return result
def _memory_event_weight(kind: str) -> int:
    weights = {
        "user_message": 9,
        "files_changed": 9,
        "error": 9,
        "approval_request": 7,
        "approval_resolved": 6,
        "tool_result": 5,
        "assistant_message": 4,
        "tool_call": 3,
    }
    return weights.get(kind, 1)


def _dedupe_memory_items(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        compact = " ".join(item.split())
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        result.append(compact)
    return result


def _fit_memory_snapshot(snapshot: dict[str, list[str]], budget: int) -> dict[str, list[str]]:
    fitted = {"relevant": [], "recent": [], "files_touched": []}
    used = 0
    for key in ["relevant", "files_touched", "recent"]:
        for item in snapshot[key]:
            cost = len(item) + 4
            if used + cost > budget:
                break
            fitted[key].append(item)
            used += cost
    return fitted


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



def project_map(limit: int = 80) -> dict[str, object]:
    """Build a compact architectural map for large-project understanding."""
    files = _project_file_records(limit=max(limit, 1))
    languages = Counter(str(item["language"]) for item in files if item.get("language") != "other")
    frameworks = _detect_frameworks(files)
    edges = _dependency_edges(files)
    entry_points = [item for item in files if item.get("role") == "entrypoint"][:12]
    large_files = [item for item in files if item.get("large")][:12]
    important = sorted(files, key=_importance_score, reverse=True)[: min(limit, 20)]
    return {
        "root": str(settings.workdir_path),
        "file_count": len(files),
        "languages": dict(languages.most_common()),
        "frameworks": frameworks,
        "entry_points": _public_file_records(entry_points),
        "important_files": _public_file_records(important),
        "large_files": _public_file_records(large_files),
        "dependency_edges": edges[:80],
        "notes": [
            "Use this project map before broad edits to understand likely entry points and related files.",
            "Use read_file_range for large files instead of loading the whole file.",
            "Dependency edges are best-effort local import/link guesses, not a full compiler graph.",
        ],
    }


def project_map_context(query: str = "", budget: int = 2600) -> str:
    """Format project_map as a compact context block for the model."""
    mapping = project_map(limit=80)
    lines: list[str] = []
    if mapping.get("languages"):
        lines.append("Languages: " + ", ".join(f"{name}={count}" for name, count in dict(mapping["languages"]).items()))
    if mapping.get("frameworks"):
        lines.append("Framework hints: " + ", ".join(str(item) for item in mapping["frameworks"]))
    entries = mapping.get("entry_points") or []
    if entries:
        lines.append("Entry points: " + ", ".join(str(item.get("path")) for item in entries[:8]))
    important = mapping.get("important_files") or []
    if important:
        lines.append("Important files:")
        for item in important[:10]:
            symbols = ", ".join(str(value) for value in item.get("symbols", [])[:5])
            imports = ", ".join(str(value) for value in item.get("imports", [])[:5])
            detail = f"- {item.get('path')} [{item.get('language')}, {item.get('role')}, {item.get('lines')} lines]"
            if symbols:
                detail += f" symbols: {symbols}"
            if imports:
                detail += f" imports: {imports}"
            lines.append(detail)
    edges = mapping.get("dependency_edges") or []
    if edges:
        lines.append("Local dependency edges: " + "; ".join(f"{edge.get('from')} -> {edge.get('to')}" for edge in edges[:12]))
    result = "\n".join(lines)
    if len(result) > budget:
        result = result[:budget].rstrip() + "\n... [project map truncated]"
    return result


def _project_file_records(limit: int = 80) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for file in sorted(settings.workdir_path.rglob("*")):
        if not file.is_file() or any(part in IGNORED_DIRS for part in file.parts):
            continue
        rel = file.relative_to(settings.workdir_path).as_posix()
        suffix = file.suffix.lower()
        try:
            size = file.stat().st_size
        except OSError:
            continue
        text = ""
        indexed = suffix in TEXT_EXTENSIONS and size <= 1024 * 1024
        if indexed:
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        lines = text.count("\n") + (1 if text else 0)
        records.append({
            "path": rel,
            "bytes": size,
            "lines": lines,
            "language": _language_for_path(file),
            "role": _role_for_path(rel),
            "indexed": indexed,
            "large": size > 300_000 or lines > 1000,
            "symbols": _code_symbols(text)[:12] if text else [],
            "imports": _import_names(text)[:12] if text else [],
            "summary": detailed_file_summary(rel, text, max_chars=220) if text else "Large or non-text file; use range reads or file-specific tools.",
        })
    return sorted(records, key=_importance_score, reverse=True)[:limit]


def _public_file_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ["path", "language", "role", "bytes", "lines", "large", "symbols", "imports", "summary"]
    return [{key: item.get(key) for key in keys} for item in records]


def _importance_score(item: dict[str, object]) -> int:
    path = str(item.get("path") or "").lower()
    score = 0
    if item.get("role") == "entrypoint":
        score += 30
    if item.get("role") in {"config", "manifest"}:
        score += 18
    if item.get("role") == "test":
        score += 8
    score += min(len(item.get("symbols") or []), 10) * 2
    score += min(len(item.get("imports") or []), 10)
    if any(name in path for name in ["main", "app", "index", "package.json", "requirements.txt", "vite.config", "readme"]):
        score += 10
    if item.get("large"):
        score -= 4
    return score


def _language_for_path(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".jsx": "react", ".ts": "typescript", ".tsx": "react-typescript",
        ".html": "html", ".css": "css", ".json": "json", ".md": "markdown", ".txt": "text",
        ".c": "c", ".h": "c-header", ".yaml": "yaml", ".yml": "yaml",
    }.get(path.suffix.lower(), "other")


def _role_for_path(path: str) -> str:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if name in {"main.py", "app.py", "index.html", "main.tsx", "main.jsx", "index.tsx", "index.jsx"}:
        return "entrypoint"
    if name in {"package.json", "requirements.txt"}:
        return "manifest"
    if name in {"vite.config.ts", "vite.config.js", "tsconfig.json", ".env.example"}:
        return "config"
    if "test" in name or lower.startswith("tests/"):
        return "test"
    if lower.endswith(".css"):
        return "style"
    if lower.endswith(".md"):
        return "docs"
    if "/routes/" in lower or name.endswith("routes.py"):
        return "api-route"
    if lower.endswith(".html"):
        return "page"
    return "source"


def _detect_frameworks(files: list[dict[str, object]]) -> list[str]:
    hints: set[str] = set()
    paths = {str(item.get("path") or "").lower() for item in files}
    imports = " ".join(" ".join(str(value) for value in item.get("imports") or []) for item in files).lower()
    if any(path.endswith("package.json") for path in paths):
        hints.add("node/npm")
    if "react" in imports or any(path.endswith(".tsx") or path.endswith(".jsx") for path in paths):
        hints.add("react")
    if any("vite.config" in path for path in paths):
        hints.add("vite")
    if "fastapi" in imports:
        hints.add("fastapi")
    if "pydantic" in imports:
        hints.add("pydantic")
    if any(path.endswith("requirements.txt") for path in paths):
        hints.add("python requirements")
    if any(path.startswith("tests/") or "test" in path for path in paths):
        hints.add("automated tests")
    return sorted(hints)


def _dependency_edges(files: list[dict[str, object]]) -> list[dict[str, str]]:
    by_stem: dict[str, str] = {}
    for item in files:
        path = str(item.get("path") or "")
        stem = Path(path).with_suffix("").as_posix().lower()
        by_stem[stem] = path
        by_stem[Path(path).stem.lower()] = path
    edges: list[dict[str, str]] = []
    for item in files:
        source = str(item.get("path") or "")
        for raw_import in item.get("imports") or []:
            candidate = str(raw_import).replace(".", "/").replace("./", "").replace("../", "").lower().strip("/")
            target = by_stem.get(candidate) or by_stem.get(candidate.rsplit("/", 1)[-1])
            if target and target != source:
                edge = {"from": source, "to": target, "import": str(raw_import)}
                if edge not in edges:
                    edges.append(edge)
    return edges


def project_understanding(query: str = "", limit: int = 12) -> dict[str, object]:
    """Return a Claude-like orientation report for complex multi-file work.

    The report is intentionally actionable: it names the likely architecture,
    candidate files, bounded read ranges, edit risk, and verification commands
    before the agent attempts broad edits.
    """
    query = query or ""
    terms = set(_terms(query))
    mapping = project_map(limit=160)
    records = _project_file_records(limit=160)
    candidates = sorted(records, key=lambda item: _query_file_score(item, terms), reverse=True)
    if terms:
        candidates = [item for item in candidates if _query_file_score(item, terms) > 0] or candidates
    likely_files = _public_file_records(candidates[: max(1, limit)])
    large_files = _public_file_records([item for item in records if item.get("large")][:8])
    return {
        "query": query,
        "task_profile": _task_profile(query),
        "architecture": {
            "languages": mapping.get("languages") or {},
            "frameworks": mapping.get("frameworks") or [],
            "entry_points": mapping.get("entry_points") or [],
            "dependency_edges": (mapping.get("dependency_edges") or [])[:30],
        },
        "likely_files": likely_files,
        "large_files": large_files,
        "read_plan": _read_plan_for_files(candidates[: max(1, limit)], query),
        "edit_strategy": _edit_strategy(query, likely_files, large_files),
        "verification_commands": _verification_commands_for_project(records),
        "notes": [
            "Use this understanding report before large edits or broad bug fixes.",
            "Read the suggested ranges before patching; summaries are not enough for exact edits.",
            "Prefer replace_in_file for small confirmed changes and write_file for new files only.",
        ],
    }


def project_understanding_context(query: str = "", budget: int = 3600) -> str:
    report = project_understanding(query, limit=8)
    lines = ["Large-project understanding"]
    task = report.get("task_profile") or {}
    if isinstance(task, dict):
        lines.append(f"Task profile: {task.get('kind')} risk={task.get('risk')} reason={task.get('reason')}")
    arch = report.get("architecture") or {}
    if isinstance(arch, dict):
        languages = arch.get("languages") or {}
        frameworks = arch.get("frameworks") or []
        if languages:
            lines.append("Languages: " + ", ".join(f"{name}={count}" for name, count in dict(languages).items()))
        if frameworks:
            lines.append("Frameworks: " + ", ".join(str(item) for item in frameworks[:8]))
    likely = report.get("likely_files") or []
    if likely:
        lines.append("Likely files:")
        for item in likely[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('path')} [{item.get('role')}, {item.get('language')}, {item.get('lines')} lines] {item.get('summary')}")
    read_plan = report.get("read_plan") or []
    if read_plan:
        lines.append("Suggested read ranges:")
        for item in read_plan[:8]:
            if isinstance(item, dict):
                lines.append(f"- read_file_range({item.get('path')}, {item.get('start_line')}, {item.get('end_line')}) because {item.get('reason')}")
    verify = report.get("verification_commands") or []
    if verify:
        lines.append("Verification commands: " + "; ".join(str(command) for command in verify[:5]))
    result = "\n".join(lines)
    if len(result) > budget:
        result = result[:budget].rstrip() + "\n... [project understanding truncated]"
    return result


def _task_profile(query: str) -> dict[str, str]:
    lower = query.lower()
    if any(term in lower for term in ["bug", "fix", "error", "traceback", "broken", "not working"]):
        return {"kind": "bugfix", "risk": "medium", "reason": "The request asks for a focused repair that should use search, range reads, patch, and verification."}
    if any(term in lower for term in ["large", "huge", "100000", "100,000", "multi-file", "codebase"]):
        return {"kind": "large-project", "risk": "high", "reason": "The request spans broad project context and should avoid whole-file rewrites."}
    if any(term in lower for term in ["add", "implement", "build", "create", "generate", "feature"]):
        return {"kind": "feature-build", "risk": "medium", "reason": "The request likely needs a file plan, new code, and verification."}
    return {"kind": "project-question", "risk": "low", "reason": "The request appears to need project understanding before answering."}


def _query_file_score(item: dict[str, object], terms: set[str]) -> int:
    haystack = " ".join(
        [
            str(item.get("path") or ""),
            _split_identifier(str(item.get("path") or "")),
            str(item.get("role") or ""),
            str(item.get("language") or ""),
            str(item.get("summary") or ""),
            " ".join(str(value) for value in item.get("symbols") or []),
            " ".join(_split_identifier(str(value)) for value in item.get("symbols") or []),
            " ".join(str(value) for value in item.get("imports") or []),
        ]
    )
    item_terms = set(_terms(haystack))
    score = len(terms & item_terms) * 6
    score += _importance_score(item)
    if item.get("large"):
        score += 4
    return score


def _read_plan_for_files(files: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    reason = "likely relevant to the request"
    if any(term in query.lower() for term in ["bug", "fix", "error", "traceback"]):
        reason = "inspect exact code before making a focused repair"
    for item in files[:12]:
        path = str(item.get("path") or "")
        if not path:
            continue
        lines = int(item.get("lines") or 0)
        if item.get("large") or lines > 500:
            start, end = 1, min(lines or 200, 220)
            tool = "read_file_range"
        else:
            start, end = 1, min(lines or 200, 260)
            tool = "read_file" if lines and lines <= 260 else "read_file_range"
        plan.append({"path": path, "tool": tool, "start_line": start, "end_line": end, "reason": reason})
    return plan


def _edit_strategy(query: str, likely_files: list[dict[str, object]], large_files: list[dict[str, object]]) -> list[str]:
    strategy = [
        "Inspect project understanding and read-plan files before editing.",
        "Use search_files or semantic_search to confirm the exact symbol/error location.",
    ]
    if large_files:
        strategy.append("For large files, use read_file_range and replace_in_file; do not load or rewrite the whole file.")
    if likely_files:
        strategy.append("Patch the smallest confirmed block in the highest-confidence file first, then verify.")
    if any(term in query.lower() for term in ["build", "create", "implement", "feature", "add"]):
        strategy.append("For new features, create a thin vertical slice and add tests or smoke checks before broad polish.")
    return strategy


def _verification_commands_for_project(records: list[dict[str, object]]) -> list[str]:
    paths = {str(item.get("path") or "").lower() for item in records}
    commands: list[str] = []
    if "package.json" in paths or any(path.endswith("/package.json") for path in paths):
        commands.extend(["npm run build", "npm test"])
    if "requirements.txt" in paths or any(path.endswith("/requirements.txt") for path in paths):
        commands.append("python -m pytest -q")
    if any(path.endswith(".py") for path in paths):
        commands.append("python -m py_compile <changed_file.py>")
    if any(path.endswith(".html") for path in paths):
        commands.append("Open the generated/changed HTML page in the preview server")
    if not commands:
        commands.append("Run the smallest command that exercises the changed file")
    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped



def _split_identifier(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    return spaced.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")

def _event_summary(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind == "user_message":
        return "User asked: " + _short(str(event.get("content") or ""), 180)
    if kind == "assistant_message":
        return "Assistant answered: " + _short(str(event.get("content") or ""), 180)
    if kind == "tool_call":
        return f"Tool used: {event.get('name')} {_short(str(event.get('args') or ''), 140)}"
    if kind == "tool_result":
        status = "ok" if event.get("ok") else "failed"
        return f"Tool result: {event.get('name')} {status}: {_short(str(event.get('preview') or ''), 180)}"
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







