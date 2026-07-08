"""Command-line client for the local coding agent backend.

The CLI talks to the same FastAPI session API used by the web frontend, so chat
history, approvals, file edits, shell commands, and activity logs all stay in
one backend system.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_API = "http://127.0.0.1:8080"


def request(method: str, url: str, body: dict[str, Any] | None = None, timeout: int = 180) -> Any:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach backend at {url}: {exc.reason}") from exc


def api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def create_session(base: str, title: str | None = None) -> dict[str, Any]:
    return request("POST", api_url(base, "/api/sessions"), {"title": title})


def print_sessions(base: str) -> None:
    sessions = request("GET", api_url(base, "/api/sessions"))
    if not sessions:
        print("No sessions yet.")
        return
    for session in sessions:
        busy = " busy" if session.get("busy") else ""
        print(f"{session['id']}  {session['title']}{busy}")


def print_history(base: str, session_id: str) -> None:
    history = request("GET", api_url(base, f"/api/sessions/{session_id}/activity"))
    print(f"Session: {history['title']} ({history['session_id']})")
    for section in ("prompts", "tool_calls", "approvals", "file_changes", "errors"):
        rows = history.get(section) or []
        print(f"\n{section.replace('_', ' ').title()}: {len(rows)}")
        for row in rows[-10:]:
            timestamp = row.get("timestamp") or ""
            print(f"- {timestamp} {row.get('summary', '')}")


def render_events(events: list[dict[str, Any]], base: str, session_id: str) -> bool:
    """Print events. Return True when an approval was handled."""
    handled_approval = False
    for event in events:
        event_type = event.get("type")
        if event_type == "assistant_delta":
            print(event.get("content", ""), end="", flush=True)
        if event_type == "assistant_message":
            if event.get("id"):
                print()
            else:
                print(event.get("content", ""))
        elif event_type == "tool_call":
            print(f"[tool] {event.get('name')} {json.dumps(event.get('args') or {})}")
        elif event_type == "tool_result":
            status = "ok" if event.get("ok") else "error"
            preview = event.get("preview") or ""
            print(f"[tool {status}] {event.get('name')}: {preview[:500]}")
        elif event_type == "approval_request":
            approve = prompt_for_approval(event)
            response = request(
                "POST",
                api_url(base, f"/api/sessions/{session_id}/approval"),
                {"id": event["id"], "approved": approve},
            )
            handled_approval = True
            render_events(response.get("events") or [], base, session_id)
        elif event_type == "error":
            print(f"[error] {event.get('message')}", file=sys.stderr)
    return handled_approval


def prompt_for_approval(event: dict[str, Any]) -> bool:
    name = event.get("name")
    print("\nApproval required")
    if name == "run_shell":
        risk = event.get("risk")
        if risk and risk != "normal":
            print(f"Risk: {risk} - {event.get('risk_reason')}")
        print(f"Command:\n{event.get('command')}")
    else:
        print(f"File: {event.get('path')}")
        diff = event.get("diff") or ""
        print(diff[:4000])
        if len(diff) > 4000:
            print("... diff truncated in CLI")

    while True:
        answer = input("Approve? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer y or n.")


def chat(base: str, message: str, session_id: str | None, context_files: list[str]) -> str:
    if session_id is None:
        session = create_session(base, title=message[:60])
        session_id = session["id"]
        print(f"[session] {session_id}")

    response = request(
        "POST",
        api_url(base, f"/api/sessions/{session_id}/chat"),
        {"message": message, "context_files": context_files},
    )
    render_events(response.get("events") or [], base, session_id)

    while response.get("busy"):
        time.sleep(0.5)
        response = request("POST", api_url(base, f"/api/sessions/{session_id}/continue"))
        render_events(response.get("events") or [], base, session_id)
    return session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for the local coding agent backend.")
    parser.add_argument("--api", default=DEFAULT_API, help=f"Backend URL, default {DEFAULT_API}")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Send one prompt to the agent.")
    ask.add_argument("message", nargs="+", help="Prompt text.")
    ask.add_argument("--session", help="Existing session id to continue.")
    ask.add_argument("--file", action="append", default=[], help="Workspace file to attach as context.")

    sub.add_parser("sessions", help="List saved sessions.")

    history = sub.add_parser("history", help="Show categorized history for one session.")
    history.add_argument("session_id")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ask":
            chat(args.api, " ".join(args.message), args.session, args.file)
        elif args.command == "sessions":
            print_sessions(args.api)
        elif args.command == "history":
            print_history(args.api, args.session_id)
        else:
            parser.print_help()
            return 1
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
