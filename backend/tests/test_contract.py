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



def test_cli_direct_file_commands(backend_server: None) -> None:
    write_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "write", "cli_pytest.txt", "CLI", "direct", "write", "passed"],
        cwd=BACKEND_DIR,
        input="y\n",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert write_proc.returncode == 0, write_proc.stderr
    assert "Approve? [y/N]" in write_proc.stdout
    assert "Wrote" in write_proc.stdout
    assert "[history] recorded in web session" in write_proc.stdout

    sessions_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "sessions"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert sessions_proc.returncode == 0, sessions_proc.stderr
    assert "CLI write: cli_pytest.txt" in sessions_proc.stdout

    read_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "read", "cli_pytest.txt"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert read_proc.returncode == 0, read_proc.stderr
    assert "CLI direct write passed" in read_proc.stdout

    delete_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "delete", "cli_pytest.txt"],
        cwd=BACKEND_DIR,
        input="y\n",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert delete_proc.returncode == 0, delete_proc.stderr
    assert "Approve? [y/N]" in delete_proc.stdout
    assert "Deleted cli_pytest.txt" in delete_proc.stdout

    missing_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "read", "cli_pytest.txt"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert missing_proc.returncode == 2
    assert "HTTP 404" in missing_proc.stderr


def test_cli_ask_routes_obvious_file_create(backend_server: None) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "cli.py",
            "--api",
            API_BASE,
            "ask",
            "Create a file named cli_ask_pytest.txt in the workspace with the text: CLI ask route passed.",
        ],
        cwd=BACKEND_DIR,
        input="y\n",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Approve? [y/N]" in proc.stdout
    assert "Wrote" in proc.stdout

    read_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "read", "cli_ask_pytest.txt"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert read_proc.returncode == 0, read_proc.stderr
    assert "CLI ask route passed." in read_proc.stdout



def test_preview_server_start_status_and_stop(backend_server: None) -> None:
    state = request("POST", "/api/preview/start", {"path": "sample.py", "port": 19090})
    assert state["running"] is True
    assert state["port"] == 19090
    assert state["url"] == "http://127.0.0.1:19090/sample.py"

    status = request("GET", "/api/preview/status?path=sample.py&port=19090")
    assert status["running"] is True
    assert status["url"] == "http://127.0.0.1:19090/sample.py"

    stopped = request("POST", "/api/preview/stop")
    assert stopped["message"] in {"Managed preview server stopped.", "No managed preview server was running."}

def test_general_planner_routes_unknown_game_to_json_fallback() -> None:
    from app.routes.tasks import build_plan

    plan = build_plan("make a snake game")
    assert plan.mode == "json-generator-fallback"
    assert plan.suggested_files == []
    assert "JSON file generator" in " ".join(plan.notes)


def test_generated_file_validation_rejects_unsafe_paths() -> None:
    from app.routes.tasks import _validated_generated_files, TaskRunStep

    steps: list[TaskRunStep] = []
    files = _validated_generated_files(
        {
            "files": [
                {"path": "snake.html", "content": "<html><script></script></html>"},
                {"path": "../escape.py", "content": "print('bad')"},
                {"path": "binary.exe", "content": "bad"},
            ]
        },
        steps,
    )

    assert files == [{"path": "snake.html", "content": "<html><script></script></html>"}]
    assert any("outside the workspace" in step.detail or "unsupported" in step.detail for step in steps)

def test_autonomous_task_run_creates_and_verifies_game(backend_server: None, test_env: dict[str, str]) -> None:
    workspace = Path(test_env["AGENT_WORKDIR"])
    target = workspace / "bouncing_ball.html"
    if target.exists():
        target.unlink()

    result = request("POST", "/api/tasks/run", {"prompt": "make a bouncing ball game", "preview": False})

    assert result["ok"] is True
    assert result["mode"] == "game-generator"
    assert result["created_files"] == ["bouncing_ball.html"]
    assert result["preview_url"] is None
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "<canvas" in content
    assert "requestAnimationFrame" in content
    assert any(step["name"] == "verify" and step["status"] == "ok" for step in result["steps"])

def test_task_plan_api_and_cli(backend_server: None) -> None:
    plan = request("POST", "/api/tasks/plan", {"prompt": "make a bouncing ball game"})
    assert plan["mode"] == "game-generator"
    assert plan["suggested_files"][0]["path"] == "bouncing_ball.html"
    assert "requestAnimationFrame" in plan["suggested_files"][0]["content"]

    proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "task", "make", "a", "bouncing", "ball", "game"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Mode: game-generator" in proc.stdout
    assert "bouncing_ball.html" in proc.stdout


def test_cli_index_range_and_patch_commands(backend_server: None, test_env: dict[str, str]) -> None:
    workspace = Path(test_env["AGENT_WORKDIR"])
    patch_target = workspace / "patch_target.py"
    patch_target.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    index_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "index"],
        cwd=BACKEND_DIR,
        env=test_env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert index_proc.returncode == 0, index_proc.stderr
    assert "sample.py" in index_proc.stdout
    assert "patch_target.py" in index_proc.stdout
    assert "[history] recorded in web session" in index_proc.stdout

    range_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "range", "sample.py", "1", "2"],
        cwd=BACKEND_DIR,
        env=test_env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert range_proc.returncode == 0, range_proc.stderr
    assert "1: def add" in range_proc.stdout
    assert "2:     return a + b" in range_proc.stdout

    patch_proc = subprocess.run(
        [sys.executable, "cli.py", "--api", API_BASE, "patch", "patch_target.py", "return 'hello'", "return 'hi'"],
        cwd=BACKEND_DIR,
        env=test_env,
        input="y\n",
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert patch_proc.returncode == 0, patch_proc.stderr
    assert "Approve? [y/N]" in patch_proc.stdout
    assert "Patched 'patch_target.py'" in patch_proc.stdout
    assert "return 'hi'" in patch_target.read_text(encoding="utf-8")

def test_implicit_code_fence_path_extraction() -> None:
    from app.agent.loop import _extract_candidate_file_path

    assert _extract_candidate_file_path("**File: `division.py`**") == "division.py"
    assert _extract_candidate_file_path("", "create a new file called game.py") == "game.py"


def test_implicit_code_fence_becomes_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from app.agent.loop import AgentSession

    session = AgentSession(title="implicit edit test")
    session._last_user_message = "create a new file called division.py which can divide numbers"
    content = """Sure.\n\n**File: `division.py`**\n\n```python\ndef divide(a, b):\n    return a / b\n```\n"""
    captured: dict[str, object] = {}

    async def fake_execute_tool(name: str, args: dict[str, object]) -> str:
        captured["name"] = name
        captured["args"] = args
        return "ok"

    monkeypatch.setattr(session, "_execute_tool", fake_execute_tool)
    handled = asyncio.run(session._maybe_offer_implicit_edit(content))

    assert handled is True
    assert captured["name"] == "write_file"
    assert captured["args"] == {"path": "division.py", "content": "def divide(a, b):\n    return a / b\n"}




