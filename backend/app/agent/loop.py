"""HTTP-driven agent sessions.

Each AgentSession (one per chat session, managed by session_manager) runs
turns as background asyncio tasks and records everything that happens as an
ordered list of events. HTTP handlers start/resume the turn and then wait
until the agent either finishes or pauses on a mutating tool (write_file /
run_shell) that needs user approval; the response carries all events produced
since the last request. Approving or rejecting resumes the loop with the real
result (or the rejection), so the agent can edit a file, run the tests, and
react to failures within a single turn — no WebSocket required.

Sessions are persisted to MySQL (see app.db) at every turn end, so the full
transcript and conversation survive a backend restart.
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import ollama

from .. import db
from ..config import settings
from .prompts import SYSTEM_PROMPT
from . import tools
from .tools import MUTATING_TOOLS, TOOL_NAMES, TOOL_SCHEMAS

DEFAULT_TITLE = "New chat"
activity_log = logging.getLogger("agent.activity")


def _utcnow() -> datetime:
    # Naive UTC — MySQL DATETIME has no timezone.
    return datetime.now(timezone.utc).replace(tzinfo=None)


# qwen-class small models sometimes emit tool calls as loose JSON in the content instead of the
# structured tool_calls field (wrong wrapper tags, code fences, stray braces). This scans for any
# brace-balanced JSON object naming a known tool so the agent can still recover the intended call.
def _iter_balanced_json_objects(text: str):
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, n):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
        i += 1


def _extract_fallback_tool_call(content: str) -> Optional[dict[str, Any]]:
    if not content or "{" not in content:
        return None
    for candidate in _iter_balanced_json_objects(content):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(parsed, dict)
            and parsed.get("name") in TOOL_NAMES
            and isinstance(parsed.get("arguments", {}), dict)
        ):
            return {"name": parsed["name"], "arguments": parsed.get("arguments") or {}}
    return None


# Second line of defense: the model may skip tool calls entirely and dump the "fixed" file as a
# code fence. If a file was recently read/written, offer that fence as a write_file approval
# rather than silently losing the edit.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_largest_fence(content: str) -> Optional[str]:
    matches = [m.group(1) for m in _FENCE_RE.finditer(content)]
    return max(matches, key=len) if matches else None


class TurnStopped(Exception):
    pass


class AgentSession:
    def __init__(
        self,
        session_id: Optional[str] = None,
        title: str = DEFAULT_TITLE,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        conversation: Optional[list[dict[str, Any]]] = None,
        events: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        now = _utcnow()
        self.id = session_id or uuid.uuid4().hex
        self.title = title
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self._client = ollama.AsyncClient(host=settings.ollama_host)
        self.conversation: list[dict[str, Any]] = conversation or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        # Hydrated sessions keep their history but should follow the current prompt.
        if self.conversation and self.conversation[0].get("role") == "system":
            self.conversation[0] = {"role": "system", "content": SYSTEM_PROMPT}
        self.events: list[dict[str, Any]] = events or []
        self._delivered = 0  # index of the first event not yet sent to the client
        self._changed = asyncio.Event()
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._turn_task: Optional[asyncio.Task] = None
        self._stop_requested = False
        self._last_file_path: Optional[str] = None
        self.busy = False

    def info(self) -> dict[str, Any]:
        """Metadata for session lists — no transcript payload."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "busy": self.busy,
        }

    async def persist(self) -> None:
        await db.save_session(
            self.id, self.title, self.conversation, self.events, self.created_at, self.updated_at
        )

    # ------------------------------------------------------------ event log

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("timestamp", _utcnow().isoformat())
        event.setdefault("session_id", self.id)
        self.events.append(event)
        activity_log.info(json.dumps(event, ensure_ascii=False, default=str))
        self._changed.set()

    def drain(self) -> list[dict[str, Any]]:
        """Events produced since the last drain/state call."""
        new = self.events[self._delivered :]
        self._delivered = len(self.events)
        return new

    def full_state(self) -> list[dict[str, Any]]:
        """All events for this session (used to rebuild the UI after a page reload)."""
        self._delivered = len(self.events)
        return list(self.events)

    async def wait_for_pause(self) -> None:
        """Block until the turn finishes or stops at a pending approval."""
        while True:
            if not self.busy or self._pending_approvals:
                return
            self._changed.clear()
            if not self.busy or self._pending_approvals:
                return
            await self._changed.wait()

    # ------------------------------------------------------------------ API

    def start_turn(self, user_message: str, context_files: Optional[list[str]] = None) -> None:
        if self.busy:
            raise RuntimeError("Agent is busy with the current turn")
        if self.title == DEFAULT_TITLE:
            # Name the session after its first request so the session list is readable.
            title = " ".join(user_message.split())
            self.title = title[:57] + "…" if len(title) > 58 else title or DEFAULT_TITLE
        self.busy = True
        self._stop_requested = False
        self._turn_task = asyncio.create_task(self._run_turn(user_message, context_files or []))

    def resolve_approval(self, approval_id: str, approved: bool) -> None:
        future = self._pending_approvals.pop(approval_id, None)
        if future is not None and not future.done():
            future.set_result(approved)

    def request_stop(self) -> None:
        self._stop_requested = True
        for approval_id, future in list(self._pending_approvals.items()):
            if not future.done():
                future.set_result(False)
            self._pending_approvals.pop(approval_id, None)
        self._changed.set()

    def reset(self) -> None:
        if self.busy:
            raise RuntimeError("Stop the current turn before resetting")
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.events = []
        self._delivered = 0
        self._last_file_path = None
        self.updated_at = _utcnow()

    # ----------------------------------------------------------------- turn

    async def _run_turn(self, user_message: str, context_files: list[str]) -> None:
        self.conversation.append(
            {"role": "user", "content": self._with_file_context(user_message, context_files)}
        )
        self._emit({"type": "user_message", "content": user_message, "context_files": context_files})
        try:
            await self._run_loop()
        except TurnStopped:
            self.conversation.append(
                {"role": "assistant", "content": "(turn stopped by user before completion)"}
            )
        except ollama.ResponseError as exc:
            self._emit({"type": "error", "message": f"Ollama error: {exc.error}"})
        except Exception as exc:
            self._emit({"type": "error", "message": f"Agent error: {exc}"})
        finally:
            self.busy = False
            self.updated_at = _utcnow()
            self._emit({"type": "turn_end"})
            # Persist once per turn: pending approvals are always resolved by now,
            # so the stored transcript never contains a dangling approval card.
            await self.persist()

    def _with_file_context(self, user_message: str, context_files: list[str]) -> str:
        """Inline attached workspace files above the user's message so the model sees them
        without having to call read_file. Kept small — local models have tight contexts."""
        if not context_files:
            return user_message
        blocks = []
        for path in context_files[:5]:
            content = tools.read_file(path)
            if len(content) > settings.max_tool_output_chars:
                content = content[: settings.max_tool_output_chars] + "\n... [truncated]"
            self._last_file_path = path
            blocks.append(f"--- {path} ---\n{content}")
        return (
            "The user attached the following workspace file(s) as context:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nUser request: {user_message}"
        )

    async def _run_loop(self) -> None:
        for _ in range(settings.max_tool_iterations):
            self._check_stopped()
            content, tool_calls = await self._get_model_response()

            if not tool_calls:
                fallback = _extract_fallback_tool_call(content)
                if fallback is not None:
                    # The leaked text was a malformed tool call, not real prose to replay.
                    tool_calls = [fallback]
                    content = ""
                else:
                    self.conversation.append({"role": "assistant", "content": content})
                    self._emit({"type": "assistant_message", "content": content})
                    await self._maybe_offer_implicit_edit(content)
                    return

            self.conversation.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ],
                }
            )
            if content:
                self._emit({"type": "assistant_message", "content": content})

            for tc in tool_calls:
                self._check_stopped()
                result = await self._execute_tool(tc["name"], tc["arguments"])
                self.conversation.append({"role": "tool", "tool_name": tc["name"], "content": result})

        self._emit(
            {
                "type": "assistant_message",
                "content": "I hit the tool-call limit for this turn without finishing — "
                "ask again or break the request into smaller steps.",
            }
        )
        self.conversation.append({"role": "assistant", "content": "(hit tool-call limit)"})

    async def _get_model_response(self) -> tuple[str, list[dict[str, Any]]]:
        response = await self._client.chat(
            model=settings.model_name,
            messages=self.conversation,
            tools=TOOL_SCHEMAS,
            stream=False,
            options={"temperature": 0.2},
        )
        message = response.get("message") or {}
        content = message.get("content") or ""
        tool_calls: list[dict[str, Any]] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name in TOOL_NAMES:
                tool_calls.append({"name": name, "arguments": dict(fn.get("arguments") or {})})
        self._check_stopped()
        return content, tool_calls

    # ---------------------------------------------------------------- tools

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        call_id = uuid.uuid4().hex[:12]
        self._emit({"type": "tool_call", "id": call_id, "name": name, "args": _preview_args(name, args)})
        try:
            if name in MUTATING_TOOLS:
                result = await self._execute_with_approval(call_id, name, args)
            else:
                result = self._execute_read_only(name, args)
            ok = not result.startswith("Error")
        except TurnStopped:
            raise
        except Exception as exc:  # tool errors go back to the model, not up the stack
            result, ok = f"Error: {exc}", False
        self._emit({"type": "tool_result", "id": call_id, "name": name, "ok": ok, "preview": result[:500]})
        return result

    def _execute_read_only(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_directory":
            return tools.list_directory(args.get("path") or ".")
        if name == "read_file":
            path = args.get("path")
            if not path:
                return "Error: 'path' argument is required"
            self._last_file_path = path
            return tools.read_file(path)
        if name == "search_files":
            query = args.get("query")
            if not query:
                return "Error: 'query' argument is required"
            return tools.search_files(query, args.get("path") or ".")
        return f"Error: unknown tool '{name}'"

    async def _execute_with_approval(self, call_id: str, name: str, args: dict[str, Any]) -> str:
        if name == "run_shell":
            command = args.get("command")
            if not command:
                return "Error: 'command' argument is required"
            request = {"type": "approval_request", "id": call_id, "name": name, "command": command}
        elif name == "write_file":
            path = args.get("path")
            content = args.get("content")
            if not path or content is None:
                return "Error: write_file requires 'path' and 'content'"
            content = content if isinstance(content, str) else json.dumps(content, indent=2)
            args = {**args, "content": content}
            try:
                diff, is_new = tools.make_diff(path, content)
            except ValueError as exc:
                return f"Error: {exc}"
            self._last_file_path = path
            request = {
                "type": "approval_request",
                "id": call_id,
                "name": name,
                "path": path,
                "diff": diff,
                "is_new_file": is_new,
            }
        else:
            return f"Error: unknown tool '{name}'"

        approved = await self._wait_for_approval(call_id, request)
        if not approved:
            return (
                f"The user REJECTED this {name} call; it was not executed. Do not repeat the "
                f"same call unchanged — try a different approach or ask the user how to proceed."
            )

        if name == "run_shell":
            result = await tools.run_shell(args["command"])
            self._emit({"type": "files_changed", "paths": []})
            return result

        result = tools.write_file(args["path"], args["content"])
        self._emit({"type": "files_changed", "paths": [args["path"]]})
        return result

    async def _wait_for_approval(self, call_id: str, request: dict[str, Any]) -> bool:
        self._check_stopped()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_approvals[call_id] = future
        self._emit(request)
        approved = await future
        self._emit({"type": "approval_resolved", "id": call_id, "approved": approved})
        if self._stop_requested:
            raise TurnStopped()
        return approved

    # ------------------------------------------------------------ fallbacks

    async def _maybe_offer_implicit_edit(self, content: str) -> None:
        """If the model dumped a whole file as a code fence instead of calling write_file,
        surface it as a write_file approval so the edit isn't lost."""
        if not self._last_file_path:
            return
        fenced = _extract_largest_fence(content)
        if not fenced or not fenced.strip():
            return
        path = self._last_file_path
        try:
            current = tools.read_file(path)
        except Exception:
            current = ""
        if fenced.strip() == current.strip():
            return  # model just quoted the file back, not an edit

        # Don't append a tool message to the conversation: there's no preceding tool_calls
        # message, and some chat templates choke on orphaned tool results.
        await self._execute_tool("write_file", {"path": path, "content": fenced})

    def _check_stopped(self) -> None:
        if self._stop_requested:
            raise TurnStopped()


def _preview_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "write_file":
        content = args.get("content") or ""
        return {"path": args.get("path"), "content": f"<{len(content)} chars>"}
    return args
