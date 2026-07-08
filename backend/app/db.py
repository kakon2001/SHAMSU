"""MySQL persistence for chat sessions.

One row per session: metadata plus the full conversation and event log as JSON
blobs, rewritten at each turn end. That is plenty for a single-user tool and
keeps hydration trivial.

If MySQL is unreachable at startup (or an individual query fails mid-run) the
app keeps working — sessions simply live in memory only and vanish when the
backend restarts. Every public function here is a safe no-op without a pool.
"""

import json
import logging
import warnings
from datetime import datetime
from typing import Any, Optional

import aiomysql

from .config import settings

log = logging.getLogger("uvicorn.error")

_pool: Optional[aiomysql.Pool] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id CHAR(32) NOT NULL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    conversation LONGTEXT NOT NULL,
    events LONGTEXT NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


def is_available() -> bool:
    return _pool is not None


async def init_db() -> None:
    """Connect, create the database and table if missing. Never raises."""
    global _pool
    try:
        # The target database may not exist yet — bootstrap it on a bare connection.
        conn = await aiomysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            autocommit=True,
        )
        try:
            async with conn.cursor() as cur:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # "database exists" is the expected case
                    await cur.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
        finally:
            conn.close()

        _pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            db=settings.mysql_database,
            autocommit=True,
            minsize=1,
            maxsize=4,
        )
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # "table exists" is the expected case
                    await cur.execute(_SCHEMA)
        log.info("MySQL connected — sessions persist to database '%s'", settings.mysql_database)
    except Exception as exc:
        _pool = None
        log.warning("MySQL unavailable (%s) — sessions are memory-only for this run", exc)


async def close_db() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


async def load_sessions() -> list[dict[str, Any]]:
    """All stored sessions with conversation/events decoded, newest first."""
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT id, title, conversation, events, created_at, updated_at "
                    "FROM sessions ORDER BY updated_at DESC"
                )
                rows = await cur.fetchall()
        sessions = []
        for row in rows:
            try:
                row["conversation"] = json.loads(row["conversation"])
                row["events"] = json.loads(row["events"])
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping session %s: corrupt JSON payload", row["id"])
                continue
            sessions.append(row)
        return sessions
    except Exception as exc:
        log.warning("Failed to load sessions from MySQL: %s", exc)
        return []


async def save_session(
    session_id: str,
    title: str,
    conversation: list[dict[str, Any]],
    events: list[dict[str, Any]],
    created_at: datetime,
    updated_at: datetime,
) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO sessions (id, title, conversation, events, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE title = VALUES(title), "
                    "conversation = VALUES(conversation), events = VALUES(events), "
                    "updated_at = VALUES(updated_at)",
                    (
                        session_id,
                        title,
                        json.dumps(conversation, ensure_ascii=False, default=str),
                        json.dumps(events, ensure_ascii=False, default=str),
                        created_at,
                        updated_at,
                    ),
                )
    except Exception as exc:
        log.warning("Failed to save session %s to MySQL: %s", session_id, exc)


async def delete_session(session_id: str) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    except Exception as exc:
        log.warning("Failed to delete session %s from MySQL: %s", session_id, exc)
