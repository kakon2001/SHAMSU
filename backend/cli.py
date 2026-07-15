"""Command-line client for the local coding agent backend.

The CLI talks to the same FastAPI session API used by the web frontend, so chat
history, approvals, file edits, shell commands, and activity logs all stay in
one backend system. It also exposes deterministic file/shell commands for demo
verification when a small local model does not choose tool calls reliably.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.agent import tools


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


def query_path(path: str) -> str:
    return urllib.parse.urlencode({"path": path})


def create_session(base: str, title: str | None = None) -> dict[str, Any]:
    return request("POST", api_url(base, "/api/sessions"), {"title": title})



def log_cli_event(
    base: str,
    action: str,
    summary: str,
    *,
    target: str | None = None,
    approved: bool | None = None,
    ok: bool = True,
    preview: str = "",
    command: str | None = None,
    path: str | None = None,
    diff: str | None = None,
    is_new_file: bool = False,
    risk: str | None = None,
    risk_reason: str | None = None,
    changed_paths: list[str] | None = None,
) -> None:
    session = create_session(base, title=f"CLI {action}: {target or path or command or 'action'}"[:60])
    request(
        "POST",
        api_url(base, f"/api/sessions/{session['id']}/cli-event"),
        {
            "action": action,
            "target": target,
            "summary": summary,
            "approved": approved,
            "ok": ok,
            "preview": preview,
            "command": command,
            "path": path,
            "diff": diff,
            "is_new_file": is_new_file,
            "risk": risk,
            "risk_reason": risk_reason,
            "changed_paths": changed_paths or [],
        },
    )
    print(f"[history] recorded in web session {session['id']}")

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


def print_file_tree(base: str) -> None:
    tree = request("GET", api_url(base, "/api/files"))
    rendered = "\n".join(render_tree(tree))
    print(rendered)
    log_cli_event(base, "files", "CLI listed workspace files.", target=".", preview=rendered)


def render_tree(node: dict[str, Any], indent: str = "") -> list[str]:
    suffix = "/" if node.get("type") == "dir" else ""
    name = node.get("path") if node.get("path") == "." else node.get("name")
    lines = [f"{indent}{name}{suffix}"]
    for child in node.get("children") or []:
        lines.extend(render_tree(child, indent + "  "))
    return lines


def read_workspace_file(base: str, path: str) -> None:
    payload = request("GET", api_url(base, f"/api/files/content?{query_path(path)}"))
    content = payload.get("content", "")
    print(content)
    log_cli_event(base, "read", f"CLI read {path}.", target=path, path=path, preview=content)


def write_workspace_file(base: str, path: str, content: str) -> None:
    print("\nApproval required")
    print(f"File: {path}")
    print("New content:")
    print(content)
    approved = ask_approval()
    if not approved:
        print("Rejected. File was not written.")
        log_cli_event(base, "write", f"CLI rejected writing {path}.", target=path, approved=False, ok=False, path=path, preview="Rejected before writing.")
        return
    before = None
    try:
        before = request("GET", api_url(base, f"/api/files/content?{query_path(path)}")).get("content", "")
    except RuntimeError:
        before = None
    payload = request("PUT", api_url(base, "/api/files/content"), {"path": path, "content": content})
    message = f"Wrote {len(payload.get('content', ''))} characters to {payload.get('path')}."
    print(message)
    diff = f"New content:\n{content}" if before is None else f"Old content:\n{before}\n\nNew content:\n{content}"
    log_cli_event(base, "write", f"CLI wrote {path}.", target=path, approved=True, path=path, diff=diff, is_new_file=before is None, changed_paths=[path], preview=message)


def delete_workspace_file(base: str, path: str) -> None:
    print("\nApproval required")
    print(f"Delete file: {path}")
    approved = ask_approval()
    if not approved:
        print("Rejected. File was not deleted.")
        log_cli_event(base, "delete", f"CLI rejected deleting {path}.", target=path, approved=False, ok=False, command=f"delete {path}", preview="Rejected before deleting.")
        return
    request("DELETE", api_url(base, f"/api/files/content?{query_path(path)}"))
    message = f"Deleted {path}."
    print(message)
    log_cli_event(base, "delete", f"CLI deleted {path}.", target=path, approved=True, command=f"delete {path}", changed_paths=[path], preview=message)


def run_direct_shell(command: str, base: str = DEFAULT_API) -> None:
    analysis = tools.analyze_shell_command(command)
    if not analysis["allowed"]:
        message = f"Blocked: {analysis['reason']}"
        print(message)
        log_cli_event(base, "shell", f"CLI blocked shell command: {command}", target=command, approved=False, ok=False, command=command, risk=str(analysis.get("risk")), risk_reason=str(analysis.get("reason")), preview=message)
        return
    print("\nApproval required")
    risk = analysis.get("risk")
    if risk and risk != "normal":
        print(f"Risk: {risk} - {analysis.get('reason')}")
    print(f"Command:\n{command}")
    approved = ask_approval()
    if not approved:
        print("Rejected. Command was not run.")
        log_cli_event(base, "shell", f"CLI rejected shell command: {command}", target=command, approved=False, ok=False, command=command, risk=str(analysis.get("risk")), risk_reason=str(analysis.get("reason")), preview="Rejected before running.")
        return
    result = asyncio.run(tools.run_shell(command))
    print(result)
    log_cli_event(base, "shell", f"CLI ran shell command: {command}", target=command, approved=True, command=command, risk=str(analysis.get("risk")), risk_reason=str(analysis.get("reason")), preview=result)


def ask_approval() -> bool:
    while True:
        answer = input("Approve? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer y or n.")



def maybe_handle_direct_ask(base: str, message: str) -> bool:
    """Route obvious file/shell prompts to deterministic CLI actions."""
    text = " ".join(message.split())

    create_match = re.search(
        r"\b(?:create|make)\s+(?:a\s+)?file\s+(?:named\s+)?[`'\"]?([^`'\"\s]+)[`'\"]?.*?\b(?:text|content)\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if create_match:
        write_workspace_file(base, create_match.group(1), create_match.group(2).strip())
        return True

    edit_match = re.search(
        r"\b(?:edit|update|change|write)\s+[`'\"]?([^`'\"\s]+)[`'\"]?.*?\b(?:says?|to|text|content)\s*:?\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if edit_match and "." in edit_match.group(1):
        write_workspace_file(base, edit_match.group(1), edit_match.group(2).strip())
        return True

    delete_match = re.search(
        r"\b(?:delete|remove)\s+[`'\"]?([^`'\"\s]+)[`'\"]?(?:\s+from\s+the\s+workspace)?\.?$",
        text,
        flags=re.IGNORECASE,
    )
    if delete_match and "." in delete_match.group(1):
        delete_workspace_file(base, delete_match.group(1))
        return True

    read_match = re.search(r"\bread\s+[`'\"]?([^`'\"\s]+)[`'\"]?", text, flags=re.IGNORECASE)
    if read_match and "." in read_match.group(1):
        read_workspace_file(base, read_match.group(1))
        return True

    if re.search(r"\blist\s+(?:the\s+)?files\b", text, flags=re.IGNORECASE):
        print_file_tree(base)
        return True

    if re.search(r"\brun\s+(?:a\s+)?shell\s+command\b.*\bcurrent\s+directory\b", text, flags=re.IGNORECASE):
        run_direct_shell("pwd", base)
        return True

    return False
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

    return ask_approval()


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

    sub.add_parser("files", help="List workspace files without using the model.")

    read = sub.add_parser("read", help="Read a workspace file without using the model.")
    read.add_argument("path")

    write = sub.add_parser("write", help="Write a workspace file after CLI approval.")
    write.add_argument("path")
    write.add_argument("content", nargs="+", help="Text to write.")

    delete = sub.add_parser("delete", help="Delete a workspace file after CLI approval.")
    delete.add_argument("path")

    shell = sub.add_parser("shell", help="Run a workspace shell command after CLI approval.")
    shell.add_argument("command", nargs="+", help="Command text.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ask":
            message = " ".join(args.message)
            if not args.session and not args.file and maybe_handle_direct_ask(args.api, message):
                return 0
            chat(args.api, message, args.session, args.file)
        elif args.command == "sessions":
            print_sessions(args.api)
        elif args.command == "history":
            print_history(args.api, args.session_id)
        elif args.command == "files":
            print_file_tree(args.api)
        elif args.command == "read":
            read_workspace_file(args.api, args.path)
        elif args.command == "write":
            write_workspace_file(args.api, args.path, " ".join(args.content))
        elif args.command == "delete":
            delete_workspace_file(args.api, args.path)
        elif args.command == "shell":
            run_direct_shell(" ".join(args.command), args.api)
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




