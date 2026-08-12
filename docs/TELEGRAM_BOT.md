# SHAMSU Telegram Bot Bridge

SHAMSU can be used from Telegram while keeping each user's projects private.

## Security Model

- Telegram identifies each sender with a stable `telegram_user_id`.
- SHAMSU stores Telegram-owned sessions with owner id `telegram:<telegram_user_id>`.
- Telegram projects are saved in the database table `telegram_projects`.
- The bot talks to SHAMSU through `/api/telegram/*` using `X-Telegram-Bridge-Secret`.
- A user can list or message only projects whose `telegram_user_id` matches them.

This means if one faculty user creates 5 projects through Telegram, only those 5 projects are returned for that Telegram user. Other Telegram users get their own separate project list.

## Setup

1. Create a bot in Telegram using `@BotFather` and copy the bot token.
2. Edit `backend/.env`:

```env
TELEGRAM_BOT_TOKEN=your_botfather_token
TELEGRAM_BRIDGE_SECRET=choose-a-private-random-secret
SHAMSU_API_BASE=http://127.0.0.1:8080
```

3. Start Ollama if it is not already running.
4. Start SHAMSU backend:

```powershell
cd C:\Users\HP\Desktop\CSE327\backend
venv\Scripts\activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

5. In another terminal, start the Telegram bridge:

```powershell
cd C:\Users\HP\Desktop\CSE327\backend
venv\Scripts\activate
python telegram_bot.py
```

## Telegram Commands

- `/start` - show help
- `/new OpenBazaar` - create a private SHAMSU project
- `/projects` - list only your own projects
- `/use <project_id>` - switch active project
- Any other text - send the prompt to selected SHAMSU project

## Faculty Explanation

Telegram is a client interface, not a separate AI. SHAMSU still uses its backend, model, session history, approval workflow, and workspace tools. Telegram users are isolated by owner id, so one user's project history is not visible to another user.