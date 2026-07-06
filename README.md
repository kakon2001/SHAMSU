# Local Coding Agent — Starter

A from-scratch coding agent that runs entirely on your machine, using a
local model via Ollama. No cloud API calls, no third-party code access.

Two interfaces, sharing the same core logic (`core.py`):
- **`agent.py`** — terminal chat, the one you've already tested
- **`server.py` + `frontend/`** — web app version, chat UI in your browser

## Setup

```bash
cd local-coding-agent
python -m venv venv
venv\Scripts\activate        (Windows)   or   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
```

## 1. Start Ollama and pull a model (once)

```bash
ollama pull qwen3:1.7b
```
(Ollama usually runs automatically in the background after install — if
`ollama serve` errors with "address already in use," that's expected,
it just means it's already running.)

## 2a. Run the terminal version

```bash
python agent.py
```

## 2b. Run the web version

```bash
uvicorn server:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser. You'll see the same
agent, but as a chat UI — tool calls render as cards, and shell commands
pause with an Allow/Deny button instead of a terminal `y/N` prompt.

Everything the agent touches is still scoped to `./workspace/`, and every
prompt/tool-call/result is still logged to `activity_log.jsonl`, exactly
as before — only the interface changed.

> Switch models the same way as before: `set AGENT_MODEL=qwen3-coder:14b`
> (Windows) or `export AGENT_MODEL=qwen3-coder:14b` (Mac/Linux) before
> starting either `agent.py` or `uvicorn server:app`.

## How it works

1. Your prompt + tool definitions go to the local model.
2. The model replies with either plain text (done) or a tool call
   (e.g. `write_file`).
3. Your code executes the tool call (with an approval gate for shell
   commands) and feeds the result back to the model.
4. Repeat until the model responds with plain text instead of a tool call.

The web version does the same loop, but pauses and returns to the browser
whenever a `run_command` call needs approval, instead of blocking on a
terminal `input()`. `server.py`'s `AgentSession` class is where that
pause/resume state lives — worth reading if you want to understand the
mechanism.

That's the entire mechanism behind Claude Code, Cursor, and the
OpenAI cookbook agent — just swapped to a local model.

## Where to go from here

Roughly in order of value:

1. **Better edit tool** — replace `write_file` (full overwrite) with a
   real diff/patch-apply tool so the agent can make surgical edits to
   large files without rewriting them.
2. **Context management** — once you're working in real codebases, you'll
   hit the model's context limit. Add file-tree summaries and only pull
   in files the agent explicitly requests, rather than dumping everything.
3. **Persistent activity history / session resume** — read back
   `activity_log.jsonl` to resume a previous session. The web backend's
   `SESSIONS` dict is in-memory only right now — sessions vanish if you
   restart `uvicorn`. Swap in SQLite here when you want persistence.
4. **MCP support** — once the core loop is solid, add an MCP client so
   the agent can use external MCP tool servers, and optionally expose
   its own tools as an MCP server.
5. **Swap/upgrade models** — try `qwen3-coder:8b` or `14b` on stronger
   hardware once you know your speed/quality tradeoff.

## Known limitations of this starter

- Tool-calling reliability depends heavily on the model — small models
  sometimes mis-format tool calls, or need very explicit instructions.
- No retry/error-recovery logic yet — if a tool call fails badly, the
  agent may not gracefully recover. Add that as you go.
- Web sessions are in-memory only — restarting the server loses history.
- Single-user, single-machine — no auth, no multi-project handling yet.

 
 
 
 
