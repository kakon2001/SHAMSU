"""
Local Coding Agent - web backend.

Wraps the same core.py logic as agent.py, but exposes it over HTTP with
a pause/resume approval flow instead of a blocking terminal input().
"""

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core

app = FastAPI(title="Local Coding Agent")

SESSIONS: dict[str, "AgentSession"] = {}


class AgentSession:
    def __init__(self):
        self.messages = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        self.pending_calls: list[dict] = []
        self.awaiting_approval: dict | None = None
        self.transcript: list[dict] = []
        self.last_call_signature = None
        self.last_result_was_error = False

    def _emit(self, item: dict):
        self.transcript.append(item)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "pending_calls": self.pending_calls,
            "awaiting_approval": self.awaiting_approval,
            "transcript": self.transcript,
            "last_call_signature": list(self.last_call_signature)
            if self.last_call_signature else None,
            "last_result_was_error": self.last_result_was_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        session = cls()
        session.messages = data["messages"]
        session.pending_calls = data["pending_calls"]
        session.awaiting_approval = data["awaiting_approval"]
        session.transcript = data["transcript"]

        sig = data.get("last_call_signature")
        session.last_call_signature = tuple(sig) if sig else None
        session.last_result_was_error = data.get("last_result_was_error", False)

        return session

    def persist(self, session_id: str):
        core.save_session(session_id, self.to_dict())

    def send_user_message(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._emit({"type": "user_message", "content": text})
        core.log_event({"type": "user_prompt", "content": text})

        return self._run_until_pause()

    def approve(self, approved: bool):
        if not self.awaiting_approval:
            raise HTTPException(400, "No pending approval for this session.")

        call = self.awaiting_approval
        self.awaiting_approval = None

        name, args = core.parse_tool_call(call)

        core.log_event({
            "type": "approval_decision",
            "tool": name,
            "args": args,
            "approved": approved,
        })

        if approved:
            if name == "run_command":
                result = core.execute_command(args["command"])
            else:
                result = core.run_tool(name, args)
        else:
            result = "DENIED by user."

        self.last_result_was_error = result.startswith("ERROR") or result.startswith("DENIED")

        self._emit({
            "type": "tool_result",
            "name": name,
            "result": result,
        })

        core.log_event({
            "type": "tool_result",
            "name": name,
            "result": result,
        })

        self.messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

        return self._run_until_pause()

    def _run_until_pause(self, max_turns: int = 10):
        turns = 0

        while turns < max_turns:
            if not self.pending_calls:
                message = core.call_model(self.messages)
                self.messages.append(message)

                tool_calls = message.get("tool_calls")

                if not tool_calls:
                    if self.last_result_was_error:
                        self._emit({
                            "type": "warning",
                            "content": (
                                "The previous tool call failed or was denied. "
                                "Verify the final answer before trusting it."
                            ),
                        })

                    self._emit({
                        "type": "final",
                        "content": message["content"],
                    })

                    core.log_event({
                        "type": "final_response",
                        "content": message["content"],
                    })

                    return {
                        "status": "done",
                        "transcript": self.transcript,
                    }

                self.pending_calls = list(tool_calls)
                turns += 1

            while self.pending_calls:
                call = self.pending_calls.pop(0)
                name, args = core.parse_tool_call(call)

                self._emit({
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                })

                core.log_event({
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                })

                if core.tool_requires_approval(name):
                    self.awaiting_approval = call

                    self._emit({
                        "type": "approval_needed",
                        "name": name,
                        "args": args,
                        "description": core.describe_tool_call(name, args),
                    })

                    return {
                        "status": "awaiting_approval",
                        "transcript": self.transcript,
                    }

                result = core.run_tool(name, args)

                signature = (name, str(args))

                if result.startswith("ERROR") and signature == self.last_call_signature:
                    result += (
                        " REPEATED IDENTICAL FAILED CALL DETECTED. Do not retry "
                        "the exact same arguments again - they will fail again. "
                        "Re-read the file first, or try a different approach."
                    )

                self.last_call_signature = signature
                self.last_result_was_error = result.startswith("ERROR")

                self._emit({
                    "type": "tool_result",
                    "name": name,
                    "result": result,
                })

                core.log_event({
                    "type": "tool_result",
                    "name": name,
                    "result": result,
                })

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })

        self._emit({
            "type": "error",
            "content": "Hit max turns without finishing.",
        })

        return {
            "status": "done",
            "transcript": self.transcript,
        }


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


def get_or_create_session(session_id: str) -> AgentSession:
    if session_id in SESSIONS:
        return SESSIONS[session_id]

    saved_state = core.load_session(session_id)

    if saved_state:
        session = AgentSession.from_dict(saved_state)
    else:
        session = AgentSession()

    SESSIONS[session_id] = session
    return session


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = get_or_create_session(session_id)

    try:
        result = session.send_user_message(req.message)
    except Exception as exc:
        raise HTTPException(500, f"Agent error: {exc}") from exc

    session.persist(session_id)

    return {
        "session_id": session_id,
        **result,
    }


@app.post("/api/approve")
def approve(req: ApprovalRequest):
    session = get_or_create_session(req.session_id)

    if not session.awaiting_approval:
        raise HTTPException(400, "No pending approval for this session.")

    try:
        result = session.approve(req.approved)
    except Exception as exc:
        raise HTTPException(500, f"Agent error: {exc}") from exc

    session.persist(req.session_id)

    return {
        "session_id": req.session_id,
        **result,
    }


@app.get("/api/sessions")
def sessions():
    return {
        "sessions": core.list_sessions(),
    }


@app.get("/api/session/{session_id}")
def session_detail(session_id: str):
    saved_state = core.load_session(session_id)

    if not saved_state:
        raise HTTPException(404, "Session not found.")

    return saved_state


@app.get("/api/model")
def model_info():
    return {
        "model": core.MODEL_NAME,
        "workspace": str(core.WORKSPACE_DIR),
    }


@app.get("/api/history/search")
def search_history(q: str = ""):
    query = q.strip().lower()

    if not query:
        return {"results": []}

    results = []

    if not core.LOG_FILE.exists():
        return {"results": []}

    with open(core.LOG_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            searchable = json.dumps(event, ensure_ascii=False).lower()

            if query in searchable:
                results.append({
                    "line": line_number,
                    "timestamp": event.get("_ts"),
                    "type": event.get("type"),
                    "event": event,
                })

    return {"results": results[-50:]}


FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
