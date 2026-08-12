"""Persistence for chat sessions.

MySQL is used when it is available. If it is not available, the app falls back
to a local SQLite file so session history still survives backend restarts.
"""

import asyncio
import hashlib
import json
import logging
import secrets
import sqlite3
import uuid
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
    updated_at DATETIME NOT NULL,
    owner_id VARCHAR(64) NOT NULL DEFAULT 'local'
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT NOT NULL PRIMARY KEY,
    title TEXT NOT NULL,
    conversation TEXT NOT NULL,
    events TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'local'
)
"""

_MYSQL_USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id CHAR(32) NOT NULL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

_MYSQL_TOKENS_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_tokens (
    token CHAR(64) NOT NULL PRIMARY KEY,
    user_id CHAR(32) NOT NULL,
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    INDEX (user_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

_SQLITE_USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT NOT NULL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_SQLITE_TOKENS_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_tokens (
    token TEXT NOT NULL PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""

_MYSQL_TELEGRAM_PROJECTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_projects (
    id CHAR(32) NOT NULL PRIMARY KEY,
    telegram_user_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    session_id CHAR(32) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX (telegram_user_id),
    INDEX (session_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

_SQLITE_TELEGRAM_PROJECTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_projects (
    id TEXT NOT NULL PRIMARY KEY,
    telegram_user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    session_id TEXT NOT NULL,
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
        conn.execute(_SQLITE_USERS_SCHEMA)
        conn.execute(_SQLITE_TOKENS_SCHEMA)
        conn.execute(_SQLITE_TELEGRAM_PROJECTS_SCHEMA)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_telegram_projects_user ON telegram_projects(telegram_user_id)")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")


def _load_sqlite_sync() -> list[dict[str, Any]]:
    with _sqlite_connect() as conn:
        rows = conn.execute(
            "SELECT id, title, conversation, events, created_at, updated_at, owner_id "
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
    owner_id: str = "local",
) -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, conversation, events, created_at, updated_at, owner_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title = excluded.title, "
            "conversation = excluded.conversation, events = excluded.events, "
            "updated_at = excluded.updated_at, owner_id = excluded.owner_id",
            (
                session_id,
                title,
                json.dumps(conversation, ensure_ascii=False, default=str),
                json.dumps(events, ensure_ascii=False, default=str),
                created_at.isoformat(),
                updated_at.isoformat(),
                owner_id,
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
                    await cur.execute(_MYSQL_USERS_SCHEMA)
                    await cur.execute(_MYSQL_TOKENS_SCHEMA)
                    await cur.execute(_MYSQL_TELEGRAM_PROJECTS_SCHEMA)
                    try:
                        await cur.execute("ALTER TABLE sessions ADD COLUMN owner_id VARCHAR(64) NOT NULL DEFAULT 'local'")
                    except Exception:
                        pass
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
                    "SELECT id, title, conversation, events, created_at, updated_at, owner_id "
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
    owner_id: str = "local",
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
                    owner_id,
                )
            except Exception as exc:
                log.warning("Failed to save session %s to SQLite: %s", session_id, exc)
        return

    try:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO sessions (id, title, conversation, events, created_at, updated_at, owner_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE title = VALUES(title), "
                    "conversation = VALUES(conversation), events = VALUES(events), "
                    "updated_at = VALUES(updated_at), owner_id = VALUES(owner_id)",
                    (
                        session_id,
                        title,
                        json.dumps(conversation, ensure_ascii=False, default=str),
                        json.dumps(events, ensure_ascii=False, default=str),
                        created_at,
                        updated_at,
                        owner_id,
                    ),
                )
    except Exception as exc:
        log.warning("Failed to save session %s to MySQL: %s", session_id, exc)
def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    expected = _hash_password(password, salt).split("$", 2)[-1]
    return secrets.compare_digest(expected, digest)


def _sqlite_create_user_sync(email: str, password: str, name: str) -> dict[str, Any]:
    user_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    display_name = name or email.split("@", 1)[0]
    try:
        with _sqlite_connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, display_name, _hash_password(password), now),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Email is already registered") from exc
    return {"id": user_id, "email": email, "name": display_name}


def _sqlite_get_user_by_email_sync(email: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute("SELECT id, email, name, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def _sqlite_create_token_sync(user_id: str, token: str, expires_at: datetime) -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, datetime.utcnow().isoformat(), expires_at.isoformat()),
        )


def _sqlite_get_user_by_token_sync(token: str) -> dict[str, Any] | None:
    now = datetime.utcnow().isoformat()
    with _sqlite_connect() as conn:
        row = conn.execute(
            "SELECT users.id, users.email, users.name FROM auth_tokens JOIN users ON users.id = auth_tokens.user_id WHERE auth_tokens.token = ? AND auth_tokens.expires_at > ?",
            (token, now),
        ).fetchone()
    return dict(row) if row else None


def _sqlite_revoke_token_sync(token: str) -> None:
    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


async def create_user(email: str, password: str, name: str = "") -> dict[str, Any]:
    if _pool is None:
        if not _sqlite_ready:
            raise RuntimeError("Database is not available")
        return await asyncio.to_thread(_sqlite_create_user_sync, email, password, name)
    user_id = uuid.uuid4().hex
    display_name = name or email.split("@", 1)[0]
    try:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, email, display_name, _hash_password(password), datetime.utcnow()),
                )
    except Exception as exc:
        raise ValueError("Email is already registered") from exc
    return {"id": user_id, "email": email, "name": display_name}


async def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    if _pool is None:
        if not _sqlite_ready:
            return None
        row = await asyncio.to_thread(_sqlite_get_user_by_email_sync, email)
    else:
        async with _pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT id, email, name, password_hash FROM users WHERE email = %s", (email,))
                row = await cur.fetchone()
    if not row or not _verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


async def create_auth_token(user_id: str, expires_at: datetime) -> str:
    token = secrets.token_hex(32)
    if _pool is None:
        if not _sqlite_ready:
            raise RuntimeError("Database is not available")
        await asyncio.to_thread(_sqlite_create_token_sync, user_id, token, expires_at)
        return token
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO auth_tokens (token, user_id, created_at, expires_at) VALUES (%s, %s, %s, %s)",
                (token, user_id, datetime.utcnow(), expires_at),
            )
    return token


async def get_user_by_token(token: str) -> dict[str, Any] | None:
    if _pool is None:
        if not _sqlite_ready:
            return None
        return await asyncio.to_thread(_sqlite_get_user_by_token_sync, token)
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT users.id, users.email, users.name FROM auth_tokens JOIN users ON users.id = auth_tokens.user_id WHERE auth_tokens.token = %s AND auth_tokens.expires_at > %s",
                (token, datetime.utcnow()),
            )
            return await cur.fetchone()


async def revoke_auth_token(token: str) -> None:
    if _pool is None:
        if _sqlite_ready:
            await asyncio.to_thread(_sqlite_revoke_token_sync, token)
        return
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM auth_tokens WHERE token = %s", (token,))

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

def _telegram_project_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": str(data["id"]),
        "telegram_user_id": str(data["telegram_user_id"]),
        "title": str(data["title"]),
        "session_id": str(data["session_id"]),
        "created_at": _parse_datetime(data["created_at"]),
        "updated_at": _parse_datetime(data["updated_at"]),
    }


def _sqlite_create_telegram_project_sync(telegram_user_id: str, title: str, session_id: str) -> dict[str, Any]:
    project_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    with _sqlite_connect() as conn:
        conn.execute(
            "INSERT INTO telegram_projects (id, telegram_user_id, title, session_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, telegram_user_id, title, session_id, now, now),
        )
        row = conn.execute("SELECT * FROM telegram_projects WHERE id = ?", (project_id,)).fetchone()
    return _telegram_project_row(row)


def _sqlite_list_telegram_projects_sync(telegram_user_id: str) -> list[dict[str, Any]]:
    with _sqlite_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM telegram_projects WHERE telegram_user_id = ? ORDER BY updated_at DESC",
            (telegram_user_id,),
        ).fetchall()
    return [_telegram_project_row(row) for row in rows]


def _sqlite_get_telegram_project_sync(project_id: str, telegram_user_id: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute(
            "SELECT * FROM telegram_projects WHERE id = ? AND telegram_user_id = ?",
            (project_id, telegram_user_id),
        ).fetchone()
    return _telegram_project_row(row) if row else None


def _sqlite_touch_telegram_project_sync(project_id: str) -> None:
    with _sqlite_connect() as conn:
        conn.execute("UPDATE telegram_projects SET updated_at = ? WHERE id = ?", (datetime.utcnow().isoformat(), project_id))


async def create_telegram_project(telegram_user_id: str, title: str, session_id: str) -> dict[str, Any]:
    telegram_user_id = str(telegram_user_id).strip()
    safe_title = title.strip()[:255] or "Telegram project"
    if _pool is None:
        if not _sqlite_ready:
            raise RuntimeError("Database is not available")
        return await asyncio.to_thread(_sqlite_create_telegram_project_sync, telegram_user_id, safe_title, session_id)
    project_id = uuid.uuid4().hex
    now = datetime.utcnow()
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "INSERT INTO telegram_projects (id, telegram_user_id, title, session_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (project_id, telegram_user_id, safe_title, session_id, now, now),
            )
            await cur.execute("SELECT * FROM telegram_projects WHERE id = %s", (project_id,))
            row = await cur.fetchone()
    return _telegram_project_row(row)


async def list_telegram_projects(telegram_user_id: str) -> list[dict[str, Any]]:
    telegram_user_id = str(telegram_user_id).strip()
    if _pool is None:
        if not _sqlite_ready:
            return []
        return await asyncio.to_thread(_sqlite_list_telegram_projects_sync, telegram_user_id)
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM telegram_projects WHERE telegram_user_id = %s ORDER BY updated_at DESC",
                (telegram_user_id,),
            )
            rows = await cur.fetchall()
    return [_telegram_project_row(row) for row in rows]


async def get_telegram_project(project_id: str, telegram_user_id: str) -> dict[str, Any] | None:
    telegram_user_id = str(telegram_user_id).strip()
    if _pool is None:
        if not _sqlite_ready:
            return None
        return await asyncio.to_thread(_sqlite_get_telegram_project_sync, project_id, telegram_user_id)
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM telegram_projects WHERE id = %s AND telegram_user_id = %s",
                (project_id, telegram_user_id),
            )
            row = await cur.fetchone()
    return _telegram_project_row(row) if row else None


async def touch_telegram_project(project_id: str) -> None:
    if _pool is None:
        if _sqlite_ready:
            await asyncio.to_thread(_sqlite_touch_telegram_project_sync, project_id)
        return
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("UPDATE telegram_projects SET updated_at = %s WHERE id = %s", (datetime.utcnow(), project_id))
