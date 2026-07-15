# Infinite Context Upgrade

This project uses a local, privacy-preserving context layer to make small Ollama models work with larger projects and longer conversations.

## What It Does

1. Chunks workspace and uploaded text files into searchable local context.
2. Builds compact summaries for every indexed workspace file.
3. Gives uploaded files priority in context summaries.
4. Builds long-session memory from previous prompts, assistant answers, tool calls, approvals, file changes, and errors.
5. Injects a compact context packet into each agent turn:
   - previous conversation memory
   - attached file contents
   - compact project/upload summaries
   - exact matching file chunks

## Why This Counts As Better Context Handling

The model does not need the whole project pasted into the prompt. Instead, the backend retrieves and compresses the most useful context locally, then sends only the relevant summaries and chunks to the local model.

This keeps the system private because project files stay on the laptop and the model runs through Ollama.

## Important Files

- `backend/app/context_index.py`: chunking, search, file summaries, uploaded summaries, conversation memory.
- `backend/app/agent/loop.py`: injects memory and context into each user turn.
- `backend/app/routes/context.py`: exposes context summary, search, dashboard, and overview APIs.
- `backend/mcp_server.py`: exposes context tools to MCP clients.
- `backend/tests/test_contract.py`: proves the context APIs and MCP tool list work.

## Verification Commands

Run the formal tests:

```powershell
cd C:\Users\HP\Desktop\CSE327\backend
venv\Scripts\activate
python -m pytest -q
```

Expected result:

```text
6 passed
```

Run the backend and check the context overview endpoint:

```powershell
uvicorn app.main:app --reload --port 8080
```

In another terminal:

```powershell
curl.exe "http://127.0.0.1:8080/api/context/overview?query=project"
```

Expected result:

- JSON with `query`
- JSON with `overview`
- overview text containing compact file/upload summaries

Check the MCP tool list:

```powershell
'{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_server.py
```

Expected result includes:

```text
context_overview
context_summary
search_context
```

## Current Limitations

This is not yet vector search. It uses local keyword scoring, summaries, and memory compression. A future version can add embeddings or a vector database.
