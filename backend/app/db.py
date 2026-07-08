"""Persistence for chat sessions.

MySQL is used when it is available. If it is not available, the app falls back
to a local SQLite file so session history still survives backend restarts.
"""

import asyncio
import json
import logging
import sqlite3
import warnings
from datetime import datetime
from typing import Any, Optional

import aiomysql

from .config import settings

log = logging.getLogger("uvicorn.error")

_pool: Optional[aiomysql.Pool] = None
_sqlite_ready = False

_MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id CHAR(32) NOT NULL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    conversation LONGTEXT NOT NULL,
    events LONGTEXT NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT NOT NULL PRIMARY KEY,
    title TEXT NOT NULL,
    conversation TEXT NOT NULL,
    events TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def is_available() -> bool:
    return _pool is not None or _sqlite_ready


def storage_mode() -> str:
    if _pool is not None:
        return "mysql"
    if _sqlite_ready:
        return f"sqlite:{settings.history_db_file}"
    return "unavailable"


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _decode_session_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        row["conversation"] = json.loads(row["conversation"])
        row["events"] = json.loads(row["events"])
        row["created_at"] = _parse_datetime(row["created_at"])
        row["updated_at"] = _parse_datetime(row["updated_at"])
        return row
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        log.warning("Skipping session %s: corrupt stored payload (%s)", row.get("id"), exc)
        return None


def _sqlite_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.history_db_file)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sqlite_sync() -> None:
    with _sqlite_connect() as conn:
        conn.execute(_SQLITE_SCHEMA)


def _load_sqlite_sync() -> list[dict[str, Any]]:
    with _sqlite_connect() as conn:
        rows = conn.execute(
            "SELECT id, title, conversation, events, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    sessions: list[dict[str, Any]] = []
    for row in rows:
        decoded = _decode_session_row(dict(row))
        if decoded is not None:
            sessions.append(decoded)
    return sessions


def _save_sqlite_sync(
    session_id: str,
    title: str,
    conversation: list[dict[str, Any]],
    events: list[dict[str, Any]],
    created_at: datetime,
    updated_at: datetime,
) -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, conversation, events, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title = excluded.title, "
            "conversation = excluded.conversation, events = excluded.events, "
            "updated_at = excluded.updated_at",
            (
                session_id,
                title,
                json.dumps(conversation, ensure_ascii=False, default=str),
                json.dumps(events, ensure_ascii=False, default=str),
                created_at.isoformat(),
                updated_at.isoformat(),
            ),
        )


def _delete_sqlite_sync(session_id: str) -> None:
    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


async def _init_sqlite() -> None:
    global _sqlite_ready
    await asyncio.to_thread(_init_sqlite_sync)
    _sqlite_ready = True


async def init_db() -> None:
    """Connect to MySQL, or create the local SQLite fallback. Never raises."""
    global _pool, _sqlite_ready
    _sqlite_ready = False
    try:
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
                    warnings.simplefilter("ignore")
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
                    warnings.simplefilter("ignore")
                    await cur.execute(_MYSQL_SCHEMA)
        log.info("MySQL connected; sessions persist to database '%s'", settings.mysql_database)
    except Exception as exc:
        _pool = None
        log.warning("MySQL unavailable (%s); using local SQLite history", exc)
        try:
            await _init_sqlite()
            log.info("SQLite history connected at %s", settings.history_db_file)
        except Exception as sqlite_exc:
            _sqlite_ready = False
            log.error("SQLite history unavailable (%s); history will not persist", sqlite_exc)


async def close_db() -> None:
    global _pool, _sqlite_ready
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
    _sqlite_ready = False


async def load_sessions() -> list[dict[str, Any]]:
    """All stored sessions with conversation/events decoded, newest first."""
    if _pool is None:
        if _sqlite_ready:
            try:
                return await asyncio.to_thread(_load_sqlite_sync)
            except Exception as exc:
                log.warning("Failed to load sessions from SQLite: %s", exc)
        return []

    try:
        async with _pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT id, title, conversation, events, created_at, updated_at "
                    "FROM sessions ORDER BY updated_at DESC"
                )
                rows = await cur.fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            decoded = _decode_session_row(row)
            if decoded is not None:
                sessions.append(decoded)
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
        if _sqlite_ready:
            try:
                await asyncio.to_thread(
                    _save_sqlite_sync,
                    session_id,
                    title,
                    conversation,
                    events,
                    created_at,
                    updated_at,
                )
            except Exception as exc:
                log.warning("Failed to save session %s to SQLite: %s", session_id, exc)
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
        if _sqlite_ready:
            try:
                await asyncio.to_thread(_delete_sqlite_sync, session_id)
            except Exception as exc:
                log.warning("Failed to delete session %s from SQLite: %s", session_id, exc)
        return

    try:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    except Exception as exc:
        log.warning("Failed to delete session %s from MySQL: %s", session_id, exc)
