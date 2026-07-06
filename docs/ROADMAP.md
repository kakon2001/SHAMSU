# SHAMSU Roadmap

## Phase 1 — Current System Validation
- Run CLI agent
- Run web app
- Use qwen3:8b locally
- Confirm file creation works

## Phase 2 — Safety and Permission System
- Ask before write_file
- Ask before edit_file
- Ask before run_command
- Add allow/deny decision history
- Add rollback/backup before edits

## Phase 3 — Activity and Search History
- Store all prompts
- Store all tool calls
- Store all file edits
- Add session list
- Add searchable chat history
- Add login history

## Phase 4 — Context Engineering
- Scan workspace
- Summarize files
- Search relevant files
- Retrieve only useful context
- Summarize old conversation history

## Phase 5 — AST and Code Understanding
- Parse Python files
- Extract imports/functions/classes
- Detect syntax errors
- Use AST summaries in context

## Phase 6 — Harness
- Add evaluation tests
- Add benchmark prompts
- Add pass/fail scoring for tool use

## Phase 7 — MCP
- Add MCP client
- Add MCP server
- Expose local tools through MCP