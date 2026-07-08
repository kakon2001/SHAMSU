"""Minimal stdio MCP-style server for the local coding agent tools.

This implements the core JSON-RPC message flow needed to expose safe workspace
tools over stdio: initialize, tools/list, and tools/call. It deliberately wraps
the same sandboxed tool functions used by the agent.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from app.agent import tools


SERVER_INFO = {"name": "local-coding-agent-mcp", "version": "0.1.0"}


MCP_TOOLS = [
    {
        "name": "list_directory",
        "description": "List files and directories inside the sandboxed workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file inside the sandboxed workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Search workspace text files by regex or literal text.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_context",
        "description": "Search chunked workspace context for relevant snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    handlers: dict[str, Callable[[dict[str, Any]], str]] = {
        "list_directory": lambda args: tools.list_directory(args.get("path") or "."),
        "read_file": lambda args: tools.read_file(_required(args, "path")),
        "search_files": lambda args: tools.search_files(_required(args, "query"), args.get("path") or "."),
        "search_context": lambda args: tools.search_context(_required(args, "query")),
    }
    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")
    return handlers[name](arguments)


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": message_id, "result": {"tools": MCP_TOOLS}}

    if method == "tools/call":
        params = message.get("params") or {}
        try:
            result = call_tool(params.get("name") or "", params.get("arguments") or {})
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"content": [{"type": "text", "text": result}]},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            response = handle(message)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def _required(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not value:
        raise ValueError(f"Missing required argument: {key}")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
