"""
Local Coding Agent — web backend.

Wraps the same core.py logic as agent.py, but exposes it over HTTP with
a pause/resume approval flow instead of a blocking terminal input().

Run: uvicorn server:app --reload --port 8000
"""

import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core

app = FastAPI(title="Local Coding Agent")

# ---- In-memory session state --------------------------------------------
# Fine for a single local user. If you add multi-user support later,
# this is the first thing to replace with a real store.

SESSIONS: dict[str, "AgentSession"] = {}


class AgentSession:
    def __init__(self):
        self.messages = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        self.pending_calls: list[dict] = []   # tool_calls not yet processed from current turn
        self.awaiting_approval: dict | None = None  # the run_command call waiting on the user
        self.transcript: list[dict] = []      # everything for the UI to render, in order
        self.last_call_signature = None       # detects repeated identical failed tool calls
        self.last_result_was_error = False    # flags when the model claims success after a failure

    def _emit(self, item: dict):
        self.transcript.append(item)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "pending_calls": self.pending_calls,
            "awaiting_approval": self.awaiting_approval,
            "transcript": self.transcript,
            "last_call_signature": list(self.last_call_signature) if self.last_call_signature else None,
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

        if approved:
            result = core.execute_command(args["command"])
        else:
            result = "DENIED by user."

        self.last_result_was_error = result.startswith("ERROR") or result == "DENIED by user."
        self._emit({"type": "tool_result", "name": name, "result": result})
        core.log_event({"type": "tool_result", "name": name, "result": result})

        self.messages.append({
            "role": "tool", "tool_call_id": call["id"], "content": result,
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
                                "The agent's last tool call failed, but it may be "
                                "about to claim success below. Verify the result "
                                "yourself before trusting this summary."
                            ),
                        })
                    self._emit({"type": "final", "content": message["content"]})
                    core.log_event({"type": "final_response", "content": message["content"]})
                    return {"status": "done", "transcript": self.transcript}

                self.pending_calls = list(tool_calls)
                turns += 1

            while self.pending_calls:
                call = self.pending_calls.pop(0)
                name, args = core.parse_tool_call(call)
                self._emit({"type": "tool_call", "name": name, "args": args})
                core.log_event({"type": "tool_call", "name": name, "args": args})

                if name == "run_command":
                    self.awaiting_approval = call
                    self._emit({"type": "approval_needed", "command": args["command"]})
                    return {"status": "awaiting_approval", "transcript": self.transcript}

                result = core.run_tool(name, args)

                signature = (name, str(args))
                if result.startswith("ERROR") and signature == self.last_call_signature:
                    result += (
                        " REPEATED IDENTICAL FAILED CALL DETECTED. Do not retry "
                        "the exact same arguments again — they will fail again. "
                        "Re-read the file first, or try a different approach."
                    )
                self.last_call_signature = signature
                self.last_result_was_error = result.startswith("ERROR")

                self._emit({"type": "tool_result", "name": name, "result": result})
                core.log_event({"type": "tool_result", "name": name, "result": result})

                self.messages.append({
                    "role": "tool", "tool_call_id": call["id"], "content": result,
                })

        self._emit({"type": "error", "content": "Hit max turns without finishing."})
        return {"status": "done", "transcript": self.transcript}


# ---- API -------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


def get_or_create_session(session_id: str) -> AgentSession:
    """Check memory first, then fall back to the database (e.g. after a
    server restart), otherwise create a brand new session."""
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
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")
    session.persist(session_id)
    return {"session_id": session_id, **result}


@app.post("/api/approve")
def approve(req: ApprovalRequest):
    session = get_or_create_session(req.session_id)
    if not session.awaiting_approval:
        raise HTTPException(400, "No pending approval for this session.")
    try:
        result = session.approve(req.approved)
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")
    session.persist(req.session_id)
    return {"session_id": req.session_id, **result}


@app.get("/api/sessions")
def sessions():
    """List past sessions stored in the database, most recent first."""
    return {"sessions": core.list_sessions()}


@app.get("/api/model")
def model_info():
    return {"model": core.MODEL_NAME, "workspace": str(core.WORKSPACE_DIR)}


# ---- Serve the frontend ------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
