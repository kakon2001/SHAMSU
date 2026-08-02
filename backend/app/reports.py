"""Safe SQL reporting support for SHAMSU's local SQLite data.

The reporting layer is intentionally read-only for analyst queries. It can inspect
schema, execute bounded SELECT/WITH reports, and store report run metadata in a
small history table. Mutating SQL is rejected before SQLite sees it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .config import settings

MAX_LIMIT = 500
DEFAULT_LIMIT = 100
BLOCKED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|vacuum|reindex|pragma)\b",
    re.IGNORECASE,
)

_REPORT_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_history (
    id TEXT NOT NULL PRIMARY KEY,
    title TEXT NOT NULL,
    sql TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    preview_json TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


def database_path() -> Path:
    return settings.history_db_file


def ensure_reporting_tables() -> None:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(_REPORT_HISTORY_SCHEMA)
        conn.commit()


def schema_overview() -> dict[str, object]:
    ensure_reporting_tables()
    tables: list[dict[str, object]] = []
    with _read_only_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for row in rows:
            table = str(row[0])
            columns = [
                {"name": col[1], "type": col[2], "not_null": bool(col[3]), "primary_key": bool(col[5])}
                for col in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            ]
            count = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0]
            tables.append({"name": table, "columns": columns, "row_count": int(count or 0)})
    return {"database": str(database_path()), "tables": tables}


def run_report_query(sql: str, limit: int = DEFAULT_LIMIT, title: str = "", save: bool = True) -> dict[str, object]:
    ensure_reporting_tables()
    clean_sql = validate_report_sql(sql)
    bounded_limit = max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))
    wrapped_sql = f"SELECT * FROM ({clean_sql}) AS shamsu_report LIMIT ?"

    started = time.time()
    with _read_only_connection() as conn:
        cursor = conn.execute(wrapped_sql, (bounded_limit,))
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description or []]
    row_dicts = [{columns[index]: row[index] for index in range(len(columns))} for row in rows]
    result = {
        "ok": True,
        "title": title or _title_from_sql(clean_sql),
        "sql": clean_sql,
        "columns": columns,
        "rows": row_dicts,
        "row_count": len(row_dicts),
        "limit": bounded_limit,
        "elapsed_ms": round((time.time() - started) * 1000, 2),
        "saved": False,
    }
    if save:
        result["report_id"] = save_report_history(str(result["title"]), clean_sql, columns, row_dicts)
        result["saved"] = True
    return result


def validate_report_sql(sql: str) -> str:
    clean = (sql or "").strip()
    if not clean:
        raise ValueError("SQL query is required.")
    if clean.endswith(";"):
        clean = clean[:-1].strip()
    if ";" in clean:
        raise ValueError("Only one read-only SQL statement is allowed.")
    first = clean.split(None, 1)[0].lower() if clean.split(None, 1) else ""
    if first not in {"select", "with"}:
        raise ValueError("Only SELECT or WITH reporting queries are allowed.")
    if BLOCKED_SQL_RE.search(clean):
        raise ValueError("Mutating or unsafe SQL is not allowed in reporting queries.")
    return clean


def save_report_history(title: str, sql: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    ensure_reporting_tables()
    report_id = uuid.uuid4().hex
    preview = rows[:25]
    with sqlite3.connect(database_path()) as conn:
        conn.execute(
            "INSERT INTO report_history (id, title, sql, row_count, columns_json, preview_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (report_id, title, sql, len(rows), json.dumps(columns), json.dumps(preview, ensure_ascii=False, default=str), time.time()),
        )
        conn.commit()
    return report_id


def report_history(limit: int = 20) -> list[dict[str, object]]:
    ensure_reporting_tables()
    bounded = max(1, min(100, int(limit or 20)))
    with sqlite3.connect(database_path()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, title, sql, row_count, columns_json, preview_json, created_at FROM report_history ORDER BY created_at DESC LIMIT ?",
            (bounded,),
        ).fetchall()
    history = []
    for row in rows:
        history.append(
            {
                "id": row["id"],
                "title": row["title"],
                "sql": row["sql"],
                "row_count": row["row_count"],
                "columns": json.loads(row["columns_json"] or "[]"),
                "preview": json.loads(row["preview_json"] or "[]"),
                "created_at": row["created_at"],
            }
        )
    return history


def _read_only_connection() -> sqlite3.Connection:
    path = database_path()
    if not path.exists():
        ensure_reporting_tables()
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _title_from_sql(sql: str) -> str:
    compact = " ".join(sql.split())
    return compact[:80] or "SQL report"
