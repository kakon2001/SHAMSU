# Product Requirements Document: Local Coding Agent

## 1. Project Goal

Build a local Claude-like coding agent that can chat with a user, inspect workspace files, suggest code changes, ask for approval before risky actions, and store prompt/activity history. The system should run locally with an Ollama model and expose both a web interface and command-line interface.

## 2. Target Users

- Students or developers who want a local coding assistant.
- Users who need file editing support without sending project files to third-party model APIs.
- Faculty/evaluators reviewing a local agent system with backend, frontend, CLI, history, permissions, and tool support.

## 3. Core Requirements

### Local Model

- The model must run through Ollama on the user's machine.
- The default project model can be a Qwen3 model.
- The runtime model should be configurable through `backend/.env`.
- Larger models may be used when hardware allows; lighter models may be used for demo responsiveness.

### Backend

- Provide a FastAPI backend.
- Manage chat sessions.
- Call the local Ollama model.
- Provide safe tools for file and shell operations.
- Store history persistently.
- Expose APIs used by both web frontend and CLI.

### Frontend

- Provide a web application.
- Show workspace files.
- Show code editor tabs.
- Show chat messages.
- Show tool calls and approval requests.
- Show activity history.

### CLI

- Provide a terminal client.
- Allow one-shot prompts.
- Allow listing saved sessions.
- Allow viewing session history.
- Use the same backend API as the web frontend.

### Workspace Sandboxing

- The agent can only access files inside the configured workspace directory.
- File paths must be resolved and checked before read/write actions.
- Attempts to escape the workspace must be rejected.

### Permission Management

- Read-only tools can run automatically.
- Mutating tools require user approval.
- File writes must show a diff before being applied.
- Shell commands must show the command before execution.

### History

- Store conversation history.
- Store prompt history.
- Store activity history.
- Store tool call history.
- Store approval history.
- Store file-change history.
- Use MySQL if available.
- Fall back to local SQLite if MySQL is unavailable.
- Write JSON-lines activity logs.

## 4. Current Implementation Status

### Implemented

- FastAPI backend.
- React frontend.
- Ollama local model integration.
- Workspace file tree.
- Monaco editor.
- Agent chat loop.
- File tools: list, read, search, write.
- Shell command tool.
- Approval flow for file writes and shell commands.
- Diff preview for file writes.
- Multi-session support.
- Persistent history through MySQL or SQLite fallback.
- Activity log file.
- CLI client in `backend/cli.py`.
- Root README and frontend README.
- Git cleanup through `.gitignore`.
- Configurable model name and model output limits.

### Partially Implemented

- Security: workspace sandbox and approval exist, but shell command risk checks need improvement.
- Context engineering: selected file context exists, but full retrieval/chunking/summarization is not implemented.
- Performance: model limits and lightweight demo model support exist, but streaming is not implemented.

### Not Yet Implemented

- MCP server.
- Full infinite context handler.
- Formal evaluation harness.
- Streaming responses.
- Cross-platform shell execution for macOS/Linux.
- Authentication for the web app.

## 5. Functional Requirements

### Chat

- User can send a prompt.
- Agent can respond using the local model.
- Agent can use tools when needed.
- User can stop a running turn.

### File Browsing

- User can see workspace files.
- User can open files in the editor.
- User can manually save editor changes.

### File Editing by Agent

- Agent can propose file changes.
- Backend generates a diff.
- User approves or rejects the change.
- Approved changes are written to disk.

### Shell Commands

- Agent can request a shell command.
- User approves or rejects the command.
- Backend runs approved commands in the workspace directory.

### Sessions

- User can create sessions.
- User can switch sessions.
- User can delete sessions.
- User can clear a session transcript.
- Sessions persist after backend restart.

### CLI

- User can ask questions from terminal.
- User can attach a workspace file.
- User can list sessions.
- User can view categorized history.
- User can approve or reject agent actions from terminal.

## 6. Non-Functional Requirements

### Privacy

- Project files are processed locally.
- Model runs locally through Ollama.
- No external model API is required.

### Performance

- Model choice must be configurable.
- Smaller local models may be used for limited hardware.
- Output length and context size should be configurable.

### Reliability

- If MySQL is unavailable, SQLite fallback must keep history working.
- Backend should return useful errors for model or tool failures.

### Maintainability

- Backend, frontend, CLI, and docs should be separated clearly.
- Environment settings should be documented.
- Generated files should not be committed.

## 7. Security Requirements

Current security:

- Workspace path sandboxing.
- Approval before file writes.
- Approval before shell commands.
- Shell command risk labels.
- Blocking for obvious destructive/system-level shell commands.
- `.env` ignored by Git.

Planned security:

- Redact secrets from logs.
- Add authentication for frontend/API.
- Add audit viewer for approvals and tool use.

## 8. Demo Acceptance Criteria

The demo is successful if:

1. Backend starts locally.
2. Frontend opens at `http://localhost:5173`.
3. Health API shows active model and history store.
4. User can send a chat prompt.
5. User can browse workspace files.
6. Agent can explain a file.
7. Agent can propose a file edit.
8. User can approve or reject the edit.
9. Session history survives backend restart.
10. CLI can send a prompt and list sessions.

## 9. Future Work

1. Add evaluation harness.
2. Add stronger shell command security.
3. Add streaming responses.
4. Add context chunking and retrieval.
5. Add summarization for long conversations.
6. Add MCP server integration.
7. Add macOS/Linux shell support.
8. Add authentication.
