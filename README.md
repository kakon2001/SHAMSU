# Local Coding Agent

This project is a local Claude-style coding agent with a FastAPI backend and React frontend. It runs an Ollama model locally, works inside a sandboxed workspace, asks for approval before file writes or shell commands, and stores chat/activity history.

## What It Includes

- Local model integration through Ollama.
- Default model: `qwen3:8b`.
- FastAPI backend.
- React + TypeScript frontend.
- Workspace file tree and Monaco editor.
- Agent tools for listing, reading, searching, writing files, and running shell commands.
- Approval flow for file edits and shell commands.
- Session, prompt, tool, approval, file-change, and error history.
- Persistent history through MySQL when available, or local SQLite fallback.
- JSON-lines activity logging at `logs/activity.log`.

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

## Git Notes

Do not commit local environments or generated files. The root `.gitignore` excludes `.env`, virtual environments, Python cache files, frontend builds, Node dependencies, logs, and local session database files.
