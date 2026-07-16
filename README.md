# SHAMSU

This project is a local Claude-style coding agent with a FastAPI backend and React frontend. It runs an Ollama model locally, works inside a sandboxed workspace, asks for approval before file writes or shell commands, and stores chat/activity history.

## What It Includes

- Local model integration through Ollama.
- Default model: `qwen3:8b`.
- FastAPI backend.
- React + TypeScript frontend.
- Command-line client for terminal usage.
- Claude-style chat/editor layout with Monaco editor.
- Agent tools for listing, reading, searching, writing files, and running shell commands.
- Context handler for chunking, summarizing, and searching workspace/uploaded files.
- External PDF/text upload that converts documents into local searchable context.
- Minimal stdio MCP-style server for safe workspace tools.
- Cross-platform shell command runner foundation.
- Approval flow for file edits and shell commands.
- Session, prompt, tool, approval, file-change, and error history.
- Persistent history through MySQL when available, or local SQLite fallback.
- JSON-lines activity logging at `logs/activity.log`.

## Project Documents

- Product requirements: `docs/PRD.md`

## Setup

Pull the model:

```powershell
ollama pull qwen3:8b
```

Start the backend:

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Start the frontend:

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open `http://localhost:5173`.

Use the CLI after the backend is running:

```powershell
cd backend
venv\Scripts\activate
python cli.py ask "Explain the files in the workspace"
python cli.py sessions
python cli.py history <session_id>
```

Run the basic harness after the backend is running:

```powershell
cd backend
venv\Scripts\activate
python harness.py
```

## Configuration

Backend configuration is in `backend/.env`.

```env
AGENT_WORKDIR=../workspace
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=qwen3:8b
FRONTEND_ORIGIN=http://localhost:5173
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=coding_agent
HISTORY_DB_PATH=../sessions.db
ACTIVITY_LOG_PATH=../logs/activity.log
```

Frontend configuration is in `frontend/.env`.

```env
VITE_API_BASE=http://localhost:8080
```

## History API

The full session state is available at:

```text
GET /api/sessions/{session_id}/state
```

Categorized activity history is available at:

```text
GET /api/sessions/{session_id}/activity
```

The activity response separates prompts, tool calls, approvals, file changes, and errors.

If MySQL is not running, the backend automatically stores sessions in `sessions.db` at the project root. Activity events are also written to `logs/activity.log`.

## Context API

The context handler chunks workspace text files, summarizes indexed files, and searches relevant snippets. Uploaded PDF/text files are extracted into local text files under `workspace/uploads/`, which is ignored by Git.

```text
GET /api/context/summary
GET /api/context/search?query=calculator
GET /api/context/auto?query=calculator
```

The agent automatically injects a small relevant context pack into each prompt. It can also call `search_context` when it needs broader project context.

## External File Upload

The web chat has an `Upload` button for PDFs and text/code files. After upload, the backend extracts readable text, saves it locally as `workspace/uploads/...txt`, attaches that file to the next prompt, and refreshes the editor file list.

Supported upload examples:

- `.pdf`
- `.txt`, `.md`, `.csv`, `.json`
- `.py`, `.js`, `.ts`, `.tsx`
- `.html`, `.css`, `.yaml`, `.yml`, `.log`

Upload API:

```text
POST /api/uploads
```

## CLI

The CLI is implemented in `backend/cli.py`. It calls the same backend API as the web app, so it uses the same sessions and history.

Examples:

```powershell
python cli.py ask "Say hello"
python cli.py ask "Explain calculator.py" --file calculator.py
python cli.py sessions
python cli.py history <session_id>
```

If the agent asks to edit a file or run a shell command, the CLI prints the command or diff and asks for approval in the terminal.

## Harness

The basic harness is implemented in `backend/harness.py`. It checks backend health, active model reporting, history storage, session creation, activity history, context summary, automatic context retrieval, external upload, CLI availability, and session deletion.

## MCP Server

The MCP server foundation is implemented in `backend/mcp_server.py`. It exposes safe read-only workspace tools over stdio:

- `list_directory`
- `read_file`
- `search_files`
- `search_context`

Run it from the backend folder:

```powershell
python mcp_server.py
```

It accepts JSON-RPC messages on stdin for `initialize`, `tools/list`, and `tools/call`.

## Git Notes

Do not commit local environments or generated files. The root `.gitignore` excludes `.env`, virtual environments, Python cache files, frontend builds, Node dependencies, logs, and local session database files.

## MCP Server

The MCP server is implemented in `backend/mcp_server.py`. It exposes safe read-only workspace tools and resources over stdio using JSON-RPC style MCP messages.

Supported methods:

- `initialize`
- `ping`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`

Exposed tools:

- `list_directory`
- `read_file`
- `search_files`
- `search_context`
- `context_summary`

Exposed resources:

- `workspace://summary`
- `workspace://tree`
- `workspace://file/{path}`

Run it from the backend folder:

```powershell
python mcp_server.py
```

Example:

```powershell
'{"jsonrpc":"2.0","id":1,"method":"resources/list"}' | python mcp_server.py
```

