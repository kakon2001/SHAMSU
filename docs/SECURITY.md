# SHAMSU Security Model

## Principles

1. Local-first execution
2. No third-party code exposure by default
3. Workspace sandboxing
4. Ask-before-change
5. Activity logging
6. Permission-based tool execution

## File Safety

The agent can only work inside the configured workspace directory.

## Permission Rules

Read-only tools may run automatically.

Dangerous tools require user approval:
- write_file
- edit_file
- run_command
- delete_file
- apply_patch

## Logs

The system logs:
- prompts
- tool calls
- tool results
- approval decisions
- file changes
- errors

## Future Security Features

- local authentication
- login history
- rollback before edits
- secret scanning
- command risk scoring