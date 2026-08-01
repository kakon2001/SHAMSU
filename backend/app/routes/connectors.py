from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/connectors", tags=["connectors"])

CONNECTORS: list[dict[str, Any]] = [
    {
        "id": "workspace-files",
        "name": "Workspace Files",
        "category": "local",
        "status": "enabled",
        "privacy": "local-only",
        "description": "Read, search, edit, and verify files inside the sandboxed SHAMSU workspace.",
        "capabilities": ["list files", "read files", "search code", "write with approval", "patch with approval"],
        "tools": ["list_directory", "read_file", "read_file_range", "search_files", "write_file", "replace_in_file"],
        "setup_required": False,
    },
    {
        "id": "context-engine",
        "name": "Context Engine",
        "category": "local",
        "status": "enabled",
        "privacy": "local-only",
        "description": "Indexes workspace files and uploads into summaries and searchable chunks for larger-project understanding.",
        "capabilities": ["project map", "context search", "upload summaries", "large-file range reading"],
        "tools": ["search_context", "project_map", "project_index"],
        "setup_required": False,
    },
    {
        "id": "web-search",
        "name": "Web Search",
        "category": "internet",
        "status": "enabled",
        "privacy": "external-request",
        "description": "Searches the public web for current facts and returns source URLs.",
        "capabilities": ["current information", "source URLs", "external documentation lookup"],
        "tools": ["web_search"],
        "setup_required": False,
    },
    {
        "id": "git-dashboard",
        "name": "Git / GitHub Read-Only Dashboard",
        "category": "repo",
        "status": "enabled",
        "privacy": "local-repo",
        "description": "Shows local repository status, recent commits, diffs, and code/commit search.",
        "capabilities": ["git status", "commit log", "code search", "diff inspection"],
        "tools": ["git_status", "git_log", "git_search", "git_diff"],
        "setup_required": False,
    },
    {
        "id": "mcp-server",
        "name": "MCP Server",
        "category": "protocol",
        "status": "enabled",
        "privacy": "local-protocol",
        "description": "Exposes SHAMSU tools over stdio JSON-RPC/MCP-style calls for external clients.",
        "capabilities": ["tools/list", "tools/call", "resources/list", "resources/read"],
        "tools": ["mcp_server.py"],
        "setup_required": False,
    },
    {
        "id": "preview-server",
        "name": "Preview Server",
        "category": "local",
        "status": "enabled",
        "privacy": "local-only",
        "description": "Starts and stops local previews for generated HTML and web app projects.",
        "capabilities": ["start preview", "stop preview", "preview status", "open local URL"],
        "tools": ["preview_start", "preview_stop", "preview_status"],
        "setup_required": False,
    },
    {
        "id": "google-drive",
        "name": "Google Drive",
        "category": "cloud",
        "status": "planned",
        "privacy": "oauth-required",
        "description": "Future connector for Drive/Docs/PDF files with user login and scoped permissions.",
        "capabilities": ["external files", "cloud documents", "PDF/document retrieval"],
        "tools": [],
        "setup_required": True,
    },
    {
        "id": "database-connector",
        "name": "External Database Connector",
        "category": "database",
        "status": "planned",
        "privacy": "credentials-required",
        "description": "Future connector for MySQL/Postgres/Supabase project data beyond local SQLite demos.",
        "capabilities": ["schema inspect", "query", "generated app persistence"],
        "tools": [],
        "setup_required": True,
    },
]


@router.get("")
def list_connectors() -> dict[str, Any]:
    enabled = sum(1 for connector in CONNECTORS if connector["status"] == "enabled")
    planned = sum(1 for connector in CONNECTORS if connector["status"] == "planned")
    return {"connectors": CONNECTORS, "enabled_count": enabled, "planned_count": planned}


@router.get("/{connector_id}")
def get_connector(connector_id: str) -> dict[str, Any]:
    for connector in CONNECTORS:
        if connector["id"] == connector_id:
            return connector
    from fastapi import HTTPException

    raise HTTPException(status_code=404, detail="Connector not found")
