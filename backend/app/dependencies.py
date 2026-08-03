from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.@/-]+(?:==[A-Za-z0-9_.!+*-]+|>=[A-Za-z0-9_.!+*-]+|<=[A-Za-z0-9_.!+*-]+)?$")
PY_PROJECT_FILES = {"requirements.txt", "pyproject.toml"}
JS_PROJECT_FILES = {"package.json"}

LIBRARY_SUGGESTIONS: dict[str, list[dict[str, str]]] = {
    "chart": [
        {"manager": "npm", "package": "recharts", "reason": "React dashboard charts and reporting visuals."},
        {"manager": "npm", "package": "chart.js", "reason": "Canvas-based charts for plain JavaScript or lightweight dashboards."},
    ],
    "dashboard": [
        {"manager": "npm", "package": "recharts", "reason": "Dashboard metrics and trend charts."},
        {"manager": "npm", "package": "lucide-react", "reason": "Clean dashboard icons and actions."},
    ],
    "crm": [
        {"manager": "npm", "package": "react-hook-form", "reason": "Reliable forms for contacts, deals, and validation."},
        {"manager": "npm", "package": "recharts", "reason": "CRM pipeline and sales reporting charts."},
    ],
    "pdf": [
        {"manager": "pip", "package": "pypdf", "reason": "Read and extract text from PDF uploads."},
    ],
    "auth": [
        {"manager": "pip", "package": "bcrypt", "reason": "Password hashing for local authentication."},
        {"manager": "pip", "package": "PyJWT", "reason": "JWT token creation and validation."},
    ],
    "database": [
        {"manager": "pip", "package": "SQLAlchemy", "reason": "Structured ORM/database access for larger systems."},
        {"manager": "pip", "package": "alembic", "reason": "Database migrations for evolving schemas."},
    ],
    "api": [
        {"manager": "pip", "package": "httpx", "reason": "Async HTTP calls to APIs and external services."},
    ],
    "test": [
        {"manager": "pip", "package": "pytest", "reason": "Automated Python unit and integration tests."},
        {"manager": "npm", "package": "vitest", "reason": "Frontend/unit tests for Vite projects."},
    ],
    "game": [
        {"manager": "npm", "package": "kaboom", "reason": "JavaScript game engine for browser games."},
        {"manager": "npm", "package": "phaser", "reason": "Full-featured 2D browser game framework."},
    ],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scan_projects(base: Path | None = None) -> dict[str, Any]:
    root = (base or repo_root()).resolve()
    projects: list[dict[str, Any]] = []
    for directory in _candidate_dirs(root):
        info = _project_info(directory, root)
        if info:
            projects.append(info)
    return {"root": str(root), "projects": projects, "summary": _scan_summary(projects)}


def plan_dependencies(prompt: str, target: str = "auto") -> dict[str, Any]:
    scan = scan_projects()
    lower = prompt.lower()
    suggestions: list[dict[str, str]] = []
    for keyword, libs in LIBRARY_SUGGESTIONS.items():
        if keyword in lower:
            suggestions.extend(libs)
    if not suggestions:
        if any(word in lower for word in ["website", "web app", "frontend", "react"]):
            suggestions.extend(LIBRARY_SUGGESTIONS["dashboard"])
        elif any(word in lower for word in ["backend", "server", "login", "system"]):
            suggestions.extend(LIBRARY_SUGGESTIONS["database"])
        else:
            suggestions.append({"manager": "pip", "package": "pytest", "reason": "Useful default for verifying generated Python code."})
    suggestions = _dedupe_suggestions(suggestions)
    compatible = [_attach_project(item, scan["projects"], target) for item in suggestions]
    commands = [_install_command(item) for item in compatible if item.get("project_path")]
    return {
        "prompt": prompt,
        "target": target,
        "detected": scan,
        "suggestions": compatible,
        "commands": commands,
        "workflow": [
            "Detect stack from package.json, requirements.txt, and pyproject.toml.",
            "Select libraries that match the user goal and the detected stack.",
            "Ask approval before installing packages or changing dependency files.",
            "Install with npm or pip using sanitized package names only.",
            "Update dependency manifests when needed.",
            "Run build/test verification and use any errors for a repair loop.",
        ],
    }


def install_dependencies(manager: str, packages: list[str], project_path: str, approve: bool = False, update_manifest: bool = True, run_verification: bool = False) -> dict[str, Any]:
    manager = manager.strip().lower()
    if manager not in {"npm", "pip"}:
        raise ValueError("Only npm and pip dependency installs are supported.")
    safe_packages = [_validate_package(package) for package in packages]
    project_dir = _resolve_project_path(project_path)
    command = _command_parts(manager, safe_packages)
    manifest_updates: list[str] = []
    if not approve:
        return {
            "approved": False,
            "ran": False,
            "manager": manager,
            "packages": safe_packages,
            "project_path": str(project_dir),
            "command": " ".join(command),
            "message": "Approval required before SHAMSU installs external libraries.",
            "manifest_updates": manifest_updates,
            "verification": [],
        }

    started = time.time()
    proc = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, timeout=180)
    output = _truncate((proc.stdout or "") + (proc.stderr or ""), 6000)
    if update_manifest and manager == "pip" and proc.returncode == 0:
        manifest_updates = _update_requirements(project_dir, safe_packages)
    verification = _run_dependency_verification(project_dir, manager) if run_verification else []
    return {
        "approved": True,
        "ran": True,
        "ok": proc.returncode == 0 and all(item["ok"] for item in verification),
        "manager": manager,
        "packages": safe_packages,
        "project_path": str(project_dir),
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "output": output,
        "elapsed_ms": round((time.time() - started) * 1000, 2),
        "manifest_updates": manifest_updates,
        "verification": verification,
    }


def _candidate_dirs(root: Path) -> list[Path]:
    candidates = [root, root / "backend", root / "frontend", root / "workspace"]
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.name not in {".git", "node_modules", "venv", ".venv", "dist", "build", "logs"}:
            candidates.append(child)
    seen: set[Path] = set()
    clean: list[Path] = []
    for item in candidates:
        resolved = item.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            clean.append(resolved)
    return clean


def _project_info(directory: Path, root: Path) -> dict[str, Any] | None:
    files = {file.name for file in directory.iterdir() if file.is_file()}
    managers: list[str] = []
    dependencies: dict[str, list[str]] = {}
    scripts: dict[str, str] = {}
    if files & JS_PROJECT_FILES:
        managers.append("npm")
        package_json = directory / "package.json"
        try:
            package_data = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            package_data = {}
        deps = list((package_data.get("dependencies") or {}).keys()) + list((package_data.get("devDependencies") or {}).keys())
        dependencies["npm"] = sorted(set(str(dep) for dep in deps))
        scripts = {str(key): str(value) for key, value in (package_data.get("scripts") or {}).items()}
    if files & PY_PROJECT_FILES:
        managers.append("pip")
        dependencies["pip"] = _read_python_dependencies(directory)
    if not managers:
        return None
    rel = directory.relative_to(root).as_posix() if directory != root and root in directory.parents else "."
    return {"path": rel, "absolute_path": str(directory), "managers": managers, "dependencies": dependencies, "scripts": scripts}


def _read_python_dependencies(directory: Path) -> list[str]:
    req = directory / "requirements.txt"
    deps: list[str] = []
    if req.exists():
        for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                deps.append(clean)
    return sorted(set(deps))


def _scan_summary(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "No dependency manifests found yet."
    parts = []
    for project in projects:
        managers = ", ".join(project["managers"])
        parts.append(f"{project['path']} uses {managers}")
    return "; ".join(parts)


def _dedupe_suggestions(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    clean: list[dict[str, str]] = []
    for item in items:
        key = (item["manager"], item["package"].lower())
        if key not in seen:
            seen.add(key)
            clean.append(item)
    return clean[:8]


def _attach_project(item: dict[str, str], projects: list[dict[str, Any]], target: str) -> dict[str, str]:
    manager = item["manager"]
    selected = None
    if target and target != "auto":
        selected = next((project for project in projects if project["path"] == target and manager in project["managers"]), None)
    if selected is None:
        selected = next((project for project in projects if manager in project["managers"]), None)
    return {**item, "project_path": selected["path"] if selected else "", "already_installed": str(_already_installed(item, selected)).lower()}


def _already_installed(item: dict[str, str], project: dict[str, Any] | None) -> bool:
    if not project:
        return False
    deps = project.get("dependencies", {}).get(item["manager"], [])
    name = _package_base(item["package"]).lower()
    return any(_package_base(str(dep)).lower() == name for dep in deps)


def _install_command(item: dict[str, str]) -> str:
    if item["manager"] == "npm":
        return f"cd {item['project_path']} && npm install {item['package']}"
    return f"cd {item['project_path']} && python -m pip install {item['package']}"


def _validate_package(package: str) -> str:
    clean = package.strip()
    if not clean or not PACKAGE_RE.match(clean) or ".." in clean or clean.startswith(("-", "/")):
        raise ValueError(f"Unsafe package name: {package}")
    return clean


def _package_base(package: str) -> str:
    return re.split(r"==|>=|<=|~=|>|<", package.strip(), maxsplit=1)[0]


def _resolve_project_path(project_path: str) -> Path:
    root = repo_root()
    candidate = (root / (project_path or ".")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Project path is outside the SHAMSU repository.")
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError("Project path does not exist.")
    return candidate


def _command_parts(manager: str, packages: list[str]) -> list[str]:
    if not packages:
        raise ValueError("At least one package is required.")
    if manager == "npm":
        return ["npm", "install", *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def _update_requirements(project_dir: Path, packages: list[str]) -> list[str]:
    req = project_dir / "requirements.txt"
    existing = set(_read_python_dependencies(project_dir))
    added: list[str] = []
    with req.open("a", encoding="utf-8") as handle:
        for package in packages:
            if package not in existing:
                if req.exists() and req.stat().st_size > 0:
                    handle.write("\n")
                handle.write(package)
                added.append(package)
    return [f"Added {package} to {req.name}" for package in added]


def _run_dependency_verification(project_dir: Path, manager: str) -> list[dict[str, Any]]:
    commands: list[list[str]] = []
    if manager == "npm" and (project_dir / "package.json").exists():
        commands.append(["npm", "run", "build"])
    if manager == "pip":
        commands.append([sys.executable, "-m", "pip", "check"])
    results: list[dict[str, Any]] = []
    for command in commands:
        try:
            proc = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, timeout=120)
            results.append({"command": " ".join(command), "ok": proc.returncode == 0, "exit_code": proc.returncode, "output": _truncate((proc.stdout or "") + (proc.stderr or ""), 3000)})
        except Exception as exc:
            results.append({"command": " ".join(command), "ok": False, "exit_code": None, "output": str(exc)})
    return results


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"
