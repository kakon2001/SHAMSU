from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/git", tags=["git"])

REPO_ROOT = Path(__file__).resolve().parents[3]
GIT_TIMEOUT_SECONDS = 20


class GitFileStatus(BaseModel):
    code: str
    path: str


class GitCommit(BaseModel):
    hash: str
    author: str
    date: str
    subject: str


class GitStatusResponse(BaseModel):
    ok: bool
    repo_root: str
    branch: str
    remote: str
    clean: bool
    ahead: int
    behind: int
    files: list[GitFileStatus]


class GitLogResponse(BaseModel):
    commits: list[GitCommit]


class GitDiffResponse(BaseModel):
    path: str | None
    diff: str
    truncated: bool


class GitSearchHit(BaseModel):
    kind: str
    path: str | None = None
    line: int | None = None
    commit: str | None = None
    text: str


class GitSearchResponse(BaseModel):
    query: str
    hits: list[GitSearchHit]


@router.get("/status", response_model=GitStatusResponse)
def git_status() -> GitStatusResponse:
    branch = _git(["branch", "--show-current"]).strip() or "detached"
    remote = _git(["remote", "get-url", "origin"], allow_error=True).strip()
    porcelain = _git(["status", "--porcelain=v1", "--branch"])
    files: list[GitFileStatus] = []
    ahead = behind = 0
    for line in porcelain.splitlines():
        if line.startswith("##"):
            ahead, behind = _ahead_behind(line)
            continue
        if not line.strip():
            continue
        files.append(GitFileStatus(code=line[:2].strip() or "?", path=line[3:].strip()))
    return GitStatusResponse(
        ok=True,
        repo_root=str(REPO_ROOT),
        branch=branch,
        remote=remote,
        clean=not files,
        ahead=ahead,
        behind=behind,
        files=files,
    )


@router.get("/log", response_model=GitLogResponse)
def git_log(limit: int = Query(12, ge=1, le=50), query: str = "") -> GitLogResponse:
    args = ["log", f"--max-count={limit}", "--date=short", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s"]
    if query.strip():
        args.insert(1, f"--grep={query.strip()}")
    raw = _git(args, allow_error=True)
    commits = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        commits.append(GitCommit(hash=parts[0], author=parts[1], date=parts[2], subject=parts[3]))
    return GitLogResponse(commits=commits)


@router.get("/diff", response_model=GitDiffResponse)
def git_diff(path: str | None = None, staged: bool = False) -> GitDiffResponse:
    args = ["diff", "--", path] if path else ["diff"]
    if staged:
        args.insert(1, "--staged")
    diff = _git(args, allow_error=True)
    truncated = len(diff) > 20_000
    if truncated:
        diff = diff[:20_000] + "\n... [diff truncated]"
    return GitDiffResponse(path=path, diff=diff or "(no diff)", truncated=truncated)


@router.get("/search", response_model=GitSearchResponse)
def git_search(query: str = Query(..., min_length=1), limit: int = Query(30, ge=1, le=100)) -> GitSearchResponse:
    safe_query = query.strip()
    hits: list[GitSearchHit] = []

    grep = _git(["grep", "-n", "-I", "--heading", "--break", safe_query], allow_error=True)
    for line in grep.splitlines():
        if not line or line.startswith("--"):
            continue
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            hits.append(GitSearchHit(kind="file", path=parts[0], line=int(parts[1]), text=parts[2].strip()[:240]))
            if len(hits) >= limit:
                break

    if len(hits) < limit:
        log = _git(["log", f"--max-count={limit}", f"--grep={safe_query}", "--date=short", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s"], allow_error=True)
        for line in log.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                hits.append(GitSearchHit(kind="commit", commit=parts[0], text=f"{parts[2]} {parts[1]}: {parts[3]}"))
                if len(hits) >= limit:
                    break
    return GitSearchResponse(query=query, hits=hits)


def _git(args: list[str], *, allow_error: bool = False) -> str:
    if any(arg in {"push", "pull", "reset", "clean", "checkout", "switch", "commit", "merge", "rebase"} for arg in args):
        raise HTTPException(status_code=400, detail="Only read-only git commands are allowed")
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Git command timed out") from exc
    if proc.returncode != 0 and not allow_error:
        raise HTTPException(status_code=500, detail=(proc.stderr or proc.stdout or "git command failed").strip())
    return proc.stdout or ""


def _ahead_behind(line: str) -> tuple[int, int]:
    ahead = behind = 0
    marker = line.split("[", 1)[-1].rstrip("]") if "[" in line else ""
    for part in marker.split(","):
        part = part.strip()
        if part.startswith("ahead "):
            ahead = int(part.split()[1])
        elif part.startswith("behind "):
            behind = int(part.split()[1])
    return ahead, behind