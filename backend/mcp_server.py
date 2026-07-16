"""Stdio MCP server for the SHAMSU tools.

The server exposes sandboxed workspace tools and read-only resources over a
JSON-RPC/MCP-compatible stdio transport. It supports initialize, ping,
tools/list, tools/call, resources/list, and resources/read.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable
from urllib.parse import quote, unquote

from app import context_index
from app.agent import tools
from app.config import settings


SERVER_INFO = {"name": "local-coding-agent-mcp", "version": "0.2.0"}
PROTOCOL_VERSION = "2024-11-05"
JSONRPC = "2.0"


MCP_TOOLS = [
    {
        "name": "list_directory",
        "description": "List files and directories inside the sandboxed workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
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
            "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
            "required": ["query"],
        },
    },
    {
        "name": "search_context",
        "description": "Search chunked workspace and uploaded context for relevant snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
            "required": ["query"],
        },
    },
    {
        "name": "context_summary",
        "description": "Summarize indexed workspace and uploaded context files.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 30}},
        },
    },
    {
        "name": "context_overview",
        "description": "Build a compact infinite-context overview from file and upload summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "default": ""}},
        },
    },
]


STATIC_RESOURCES = [
    {
        "uri": "workspace://summary",
        "name": "Workspace context summary",
        "description": "Summary of indexed workspace and uploaded context files.",
        "mimeType": "application/json",
    },
    {
        "uri": "workspace://tree",
        "name": "Workspace file tree",
        "description": "Directory listing for the sandboxed workspace root.",
        "mimeType": "text/plain",
    },
]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    handlers: dict[str, Callable[[dict[str, Any]], str]] = {
        "list_directory": lambda args: tools.list_directory(str(args.get("path") or ".")),
        "read_file": lambda args: tools.read_file(_required(args, "path")),
        "search_files": lambda args: tools.search_files(_required(args, "query"), str(args.get("path") or ".")),
        "search_context": lambda args: context_index.format_context_results(
            _required(args, "query"), limit=_int_arg(args, "limit", 5, 1, 20)
        ),
        "context_summary": lambda args: json.dumps(
            context_index.summarize_workspace(limit=_int_arg(args, "limit", 30, 1, 100)),
            ensure_ascii=False,
            indent=2,
        ),
        "context_overview": lambda args: context_index.automatic_summary_context(str(args.get("query") or "")),
    }
    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")
    return handlers[name](arguments)


def list_resources() -> list[dict[str, Any]]:
    resources = list(STATIC_RESOURCES)
    for item in context_index.summarize_workspace(limit=200).get("files", []):
        path = str(item.get("path") or "")
        if not path:
            continue
        resources.append(
            {
                "uri": f"workspace://file/{quote(path)}",
                "name": path,
                "description": str(item.get("summary") or "Workspace file"),
                "mimeType": "text/plain",
            }
        )
    return resources


def read_resource(uri: str) -> dict[str, Any]:
    if uri == "workspace://summary":
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps(context_index.summarize_workspace(), ensure_ascii=False, indent=2),
        }
    if uri == "workspace://tree":
        return {"uri": uri, "mimeType": "text/plain", "text": tools.list_directory(".")}
    prefix = "workspace://file/"
    if uri.startswith(prefix):
        path = unquote(uri[len(prefix) :])
        return {"uri": uri, "mimeType": "text/plain", "text": tools.read_file(path)}
    raise ValueError(f"Unknown resource: {uri}")


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return _error(None, -32600, "Invalid request")

    method = message.get("method")
    message_id = message.get("id")
    is_notification = "id" not in message

    try:
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None

        if method == "initialize":
            return _result(
                message_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": SERVER_INFO,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "instructions": "Use attached/uploaded files first. Workspace resources are sandboxed locally.",
                },
            )

        if method == "ping":
            return _result(message_id, {})

        if method == "tools/list":
            return _result(message_id, {"tools": MCP_TOOLS})

        if method == "tools/call":
            params = _params(message)
            result = call_tool(str(params.get("name") or ""), _dict_arg(params.get("arguments")))
            return _result(message_id, {"content": [{"type": "text", "text": result}], "isError": False})

        if method == "resources/list":
            return _result(message_id, {"resources": list_resources()})

        if method == "resources/read":
            params = _params(message)
            content = read_resource(_required(params, "uri"))
            return _result(message_id, {"contents": [content]})

        if is_notification:
            return None
        return _error(message_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        if is_notification:
            return None
        return _error(message_id, -32000, str(exc))


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        response = _handle_line(line)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def _handle_line(line: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error(None, -32700, f"Parse error: {exc.msg}")

    if isinstance(message, list):
        if not message:
            return _error(None, -32600, "Invalid empty batch")
        responses = [response for item in message if (response := handle(item)) is not None]
        return responses or None
    return handle(message)


def _result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": message_id, "error": {"code": code, "message": message}}


def _params(message: dict[str, Any]) -> dict[str, Any]:
    return _dict_arg(message.get("params"))


def _dict_arg(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected object arguments")
    return value


def _required(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required argument: {key}")
    return str(value)


def _int_arg(args: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    value = args.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    return max(minimum, min(maximum, parsed))


if __name__ == "__main__":
    raise SystemExit(main())


