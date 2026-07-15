"""Basic evaluation harness for the local coding agent.

Run this after starting the backend. It checks the API health, session
creation, persisted activity endpoint, and CLI script availability. It is meant
for project/demo validation, not exhaustive unit testing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "http://127.0.0.1:8080"


def request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        API_BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else None


def upload_text_file(path: str, filename: str, content: str) -> Any:
    boundary = "----coding-agent-harness"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        f"{content}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(
        API_BASE + path,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))



def mcp_request(message: dict[str, Any]) -> Any:
    server_path = Path(__file__).with_name("mcp_server.py")
    proc = subprocess.run(
        [sys.executable, str(server_path)],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "MCP server failed")
    return json.loads(proc.stdout.strip())
def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return passed


def main() -> int:
    results: list[bool] = []

    try:
        health = request("GET", "/api/health")
    except (urllib.error.URLError, TimeoutError) as exc:
        check("backend health", False, str(exc))
        return 1

    results.append(check("backend health", health.get("status") == "ok"))
    results.append(check("active model reported", bool(health.get("model")), str(health.get("model"))))
    results.append(check("history store reported", bool(health.get("history_store")), str(health.get("history_store"))))

    session = request("POST", "/api/sessions", {"title": "Harness test"})
    session_id = session["id"]
    results.append(check("session created", bool(session_id), session_id))

    state = request("GET", f"/api/sessions/{session_id}/state")
    results.append(check("session state endpoint", "events" in state and "busy" in state))

    activity = request("GET", f"/api/sessions/{session_id}/activity")
    results.append(check("activity endpoint", activity.get("session_id") == session_id))

    summary = request("GET", "/api/context/summary")
    results.append(check("context summary endpoint", "chunk_count" in summary))

    auto = request("GET", "/api/context/auto?query=calculator")
    results.append(check("automatic context endpoint", "context" in auto))

    upload = upload_text_file(
        "/api/uploads",
        "harness-notes.txt",
        "Harness upload proves external text files become searchable local context.",
    )
    results.append(check("external upload endpoint", str(upload.get("path", "")).startswith("uploads/")))


    mcp_init = mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    results.append(check("mcp initialize", "serverInfo" in mcp_init.get("result", {})))

    mcp_tools = mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tool_names = {tool.get("name") for tool in mcp_tools.get("result", {}).get("tools", [])}
    results.append(check("mcp tools/list", "context_overview" in tool_names and len(tool_names) >= 6))

    mcp_resources = mcp_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    results.append(check("mcp resources/list", len(mcp_resources.get("result", {}).get("resources", [])) >= 2))

    cli_path = Path(__file__).with_name("cli.py")
    results.append(check("cli.py exists", cli_path.exists(), str(cli_path)))

    cli_result = subprocess.run(
        [sys.executable, str(cli_path), "sessions"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    results.append(check("cli sessions command", cli_result.returncode == 0))

    request("DELETE", f"/api/sessions/{session_id}")
    results.append(check("session delete", True))

    passed = all(results)
    print("\nHarness result:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())



