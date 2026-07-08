# Local Coding Agent

A local web coding agent inspired by Claude-style workflows. The app runs a local Ollama model, lets the user chat with an agent, browse and edit files inside a sandboxed workspace, approve file/shell actions, and keep session/activity history.

## Current Features

- React + TypeScript frontend with chat, file tree, editor tabs, approval cards, and activity history.
- FastAPI backend with Ollama chat integration.
- Local model default: `qwen3:8b`.
- Sandboxed file tools: list, read, search, and write files under `AGENT_WORKDIR`.
- Approval-gated shell and file-write tools.
- Session history with prompt/tool/approval/file/error events.
- Optional MySQL persistence for chat sessions and activity logs.

## Requirements

- Python 3.12 or newer.
- Node.js 20 or newer.
- Ollama running locally.
- Qwen3 8B model pulled in Ollama.
- Optional: MySQL or XAMPP MySQL for persistent history.

## Setup

Install the model:

```bash
ollama pull qwen3:8b
```

Backend:

```powershell
cd ..\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Frontend:

```powershell
cd ..\frontend
npm install
copy .env.example .env
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Configuration

Backend settings live in `backend/.env`.

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
```

If MySQL is not running, the backend still works, but sessions are memory-only until restart.

## History

The backend stores the complete event log for each session:

- user prompts
- assistant replies
- tool calls and results
- approval requests and decisions
- changed files
- errors

Use `GET /api/sessions/{session_id}/activity` for a categorized activity-history payload.

## Git Hygiene

Generated files are ignored by the root `.gitignore`, including Python cache files, virtual environments, frontend builds, Node dependencies, logs, and local environment files.

Do not commit `backend/.env`, `venv/`, `node_modules/`, `dist/`, or `__pycache__/`.
