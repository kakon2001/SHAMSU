from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
API_BASE = "http://127.0.0.1:18080"


def request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        API_BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else None


def wait_for_server(proc: subprocess.Popen[str]) -> None:
    last_error = "server did not answer"
    for _ in range(60):
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=2)
            raise RuntimeError(f"Backend exited early. stdout={stdout!r} stderr={stderr!r}")
        try:
            health = request("GET", "/api/health")
            if health.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for backend: {last_error}")


@pytest.fixture(scope="session")
def test_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("agent_contract")
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (workspace / "notes.txt").write_text(
        "This workspace is used by pytest to verify context search and summaries.\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "AGENT_WORKDIR": str(workspace),
            "HISTORY_DB_PATH": str(root / "sessions-test.db"),
            "ACTIVITY_LOG_PATH": str(root / "activity-test.log"),
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "1",
            "MODEL_NAME": "qwen3:8b",
        }
    )
    return env


@pytest.fixture(scope="session")
def backend_server(test_env: dict[str, str]) -> None:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
        ],
        cwd=BACKEND_DIR,
        env=test_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_server(proc)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_health_reports_model_and_history_store(backend_server: None) -> None:
    health = request("GET", "/api/health")

    assert health["status"] == "ok"
    assert health["model"]
    assert "sqlite" in health["history_store"] or health["history_store"] == "mysql"
    assert health["workspace"].endswith("workspace")


def test_session_create_state_activity_and_delete(backend_server: None) -> None:
    session = request("POST", "/api/sessions", {"title": "pytest session"})
    session_id = session["id"]

    assert session["title"] == "pytest session"
    assert session["busy"] is False

    state = request("GET", f"/api/sessions/{session_id}/state")
    assert state["events"] == []
    assert state["busy"] is False

    activity = request("GET", f"/api/sessions/{session_id}/activity")
    assert activity["session_id"] == session_id
    assert "prompts" in activity
    assert "tool_calls" in activity

    deleted = request("DELETE", f"/api/sessions/{session_id}")
    assert deleted == {"ok": True}


def test_context_summary_dashboard_and_search(backend_server: None) -> None:
    summary = request("GET", "/api/context/summary")
    assert "chunk_count" in summary
    assert summary["chunk_count"] >= 1

    dashboard = request("GET", "/api/context/dashboard")
    assert dashboard["file_count"] >= 1
    assert dashboard["chunk_count"] >= 1
    assert dashboard["summary_context_budget"] >= 1
    assert dashboard["conversation_memory_budget"] >= 1
    assert "largest_files" in dashboard
    assert "file_summaries" in dashboard

    overview = request("GET", "/api/context/overview?query=pytest")
    assert overview["query"] == "pytest"
    assert "sample.py" in overview["overview"] or "notes.txt" in overview["overview"]

    search = request("GET", "/api/context/search?query=pytest&limit=3")
    assert search["query"] == "pytest"
    assert isinstance(search["matches"], list)


def test_model_list_and_switch_validation(backend_server: None) -> None:
    state = request("GET", "/api/models")
    model_ids = [model["id"] for model in state["models"]]

    assert "qwen3:8b" in model_ids
    assert "qwen3:4b" in model_ids
    assert "qwen3-8k:1.7b" in model_ids

    switched = request("POST", "/api/models/current", {"model_id": "qwen3:4b"})
    assert switched["current"] == "qwen3:4b"

    health = request("GET", "/api/health")
    assert health["model"] == "qwen3:4b"


def test_mcp_tools_list(test_env: dict[str, str]) -> None:
    proc = subprocess.run(
        [sys.executable, "mcp_server.py"],
        input='{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n',
        cwd=BACKEND_DIR,
        env=test_env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    tool_names = {tool["name"] for tool in payload["result"]["tools"]}
    assert {"list_directory", "read_file", "search_files", "search_context", "context_summary", "context_overview"}.issubset(tool_names)


def test_cli_sessions_command(backend_server: None) -> None:
    proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "sessions"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "No sessions yet." in proc.stdout or "pytest" in proc.stdout or proc.stdout.strip()

