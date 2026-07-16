# Claude-Like Agent Upgrade

This project is a local Claude/ChatGPT-style coding agent. It uses a local Ollama model for reasoning, a FastAPI backend for sessions/tools/history, and a React frontend plus CLI for interaction.

## New Capabilities

### Large File Handler

The agent should not paste a 100000-line file into the model context. Instead it can inspect a bounded range with `read_file_range`.

Demo command:

```powershell
cd C:\Users\HP\Desktop\CSE327\backend
venv\Scripts\activate
python cli.py range sample.py 1 80
```

Expected result: numbered lines from the requested file range.

### Project Index

The agent can build a compact index of workspace files, sizes, line counts, symbols, and imports. This helps it understand multi-file projects before editing.

Demo command:

```powershell
python cli.py index
```

Expected result: JSON showing project files and symbol/import summaries.

### Patch-Based Editing

For existing files, the safer workflow is to replace one exact text block instead of rewriting the whole file. The CLI and agent show a diff and ask for approval before writing.

Demo command:

```powershell
python cli.py patch hello.py "return 'hello'" "return 'hi'"
```

Expected result: an approval prompt, a unified diff, and then a patched file after you type `y`.

### Reliable Task Mode

Small local models do not always call tools reliably. Task mode gives a deterministic plan, starter code, and verification commands for larger requests such as games or bug fixes.

Demo command:

```powershell
python cli.py task "make a bouncing ball game"
```

Expected result: a `game-generator` plan with a complete `bouncing_ball.html` file and commands to run it locally.

## Current Limitation

This is not yet a fully autonomous Claude Code replacement. For very large or complex projects, the safest current workflow is:

1. Use `python cli.py task "your task"` to get a plan.
2. Use `python cli.py index` to map the project.
3. Use `python cli.py range <file> <start> <end>` to inspect relevant code.
4. Use `python cli.py patch <file> <old_text> <new_text>` or the web approval flow for edits.
5. Run the suggested verification command.

The next major upgrade would be a full task executor that automatically loops through plan -> edit -> run -> verify while still asking approval for risky file and shell actions.
