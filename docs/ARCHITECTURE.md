# SHAMSU Architecture

## Components

1. Frontend
   - Browser chat UI
   - Session history
   - Approval buttons
   - Search interface

2. Backend
   - FastAPI server
   - Agent loop
   - Tool execution
   - Permission control
   - Logging
   - Session persistence

3. Local Model
   - Ollama
   - qwen3:8b for development
   - max 12B model requirement

4. Workspace
   - User files are handled inside workspace/
   - Agent cannot access files outside allowed directory

5. Database
   - SQLite for sessions
   - Prompt history
   - Activity history
   - Login history

6. Tools
   - read_file
   - write_file
   - edit_file
   - list_dir
   - run_command
   - future: apply_patch, scan_workspace, analyze_ast