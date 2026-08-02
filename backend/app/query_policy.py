from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPolicy:
    route: str
    reason: str
    use_web_search: bool = False
    use_workspace_context: bool = False
    use_uploaded_context: bool = False
    enable_tools: bool = False


WEB_KEYWORDS = {
    "latest",
    "current",
    "today",
    "recent",
    "news",
    "now",
    "this year",
    "2025",
    "2026",
    "web search",
    "search web",
    "search online",
    "internet",
    "online",
    "price",
    "version",
    "release",
    "documentation",
    "docs",
}

WORKSPACE_KEYWORDS = {
    "workspace",
    "project",
    "repo",
    "repository",
    "codebase",
    "folder",
    "directory",
    "file",
    "files",
    "all files",
    "other files",
    "compare",
    "search",
    "find in",
    "bug",
    "error",
    "fix",
    "edit",
    "change",
    "patch",
    "run",
    "test",
    "build",
    "create",
    "write",
    "delete",
}

ATTACHMENT_KEYWORDS = {
    "this file",
    "that file",
    "uploaded file",
    "attached file",
    "uploaded pdf",
    "attached pdf",
    "the pdf",
    "the document",
    "what is this file",
    "what does this file",
    "summarize this",
    "summarise this",
}

MUTATING_KEYWORDS = {
    "edit",
    "change",
    "write",
    "create",
    "delete",
    "save",
    "run",
    "test",
    "fix",
    "execute",
    "rename",
    "move",
    "apply",
    "build",
    "implement",
}


def classify_query(user_message: str, context_files: list[str] | None = None) -> QueryPolicy:
    text = _normalise(user_message)
    has_uploads = bool(context_files)

    if has_uploads and _contains_any(text, ATTACHMENT_KEYWORDS):
        return QueryPolicy(
            route="uploaded_context",
            reason="The user is asking about an attached/uploaded file, so uploaded content is highest priority.",
            use_uploaded_context=True,
            enable_tools=_contains_any(text, MUTATING_KEYWORDS | WORKSPACE_KEYWORDS),
        )

    if wants_web_search(user_message) and not is_local_project_question(user_message):
        return QueryPolicy(
            route="web_search",
            reason="The user is asking for current or external information, so public web search should be used.",
            use_web_search=True,
            enable_tools=True,
        )

    if wants_workspace_context(user_message):
        return QueryPolicy(
            route="workspace_context",
            reason="The user is asking about local project files or code, so workspace context is highest priority.",
            use_workspace_context=True,
            enable_tools=True,
        )

    if has_uploads:
        return QueryPolicy(
            route="uploaded_context",
            reason="The user attached files, so include those files as context before answering.",
            use_uploaded_context=True,
            enable_tools=False,
        )

    return QueryPolicy(
        route="general_model",
        reason="This looks like a general explanation or reasoning question, so the model can answer directly.",
        enable_tools=False,
    )


def wants_web_search(user_message: str) -> bool:
    text = _normalise(user_message)
    if "current file" in text or "current project" in text or "current workspace" in text:
        return False
    return _contains_any(text, WEB_KEYWORDS)


def wants_workspace_context(user_message: str) -> bool:
    text = _normalise(user_message)
    if "search online" in text or "web search" in text or "search web" in text:
        return False
    return _contains_any(text, WORKSPACE_KEYWORDS)


def is_local_project_question(user_message: str) -> bool:
    text = _normalise(user_message)
    local_terms = {
        "workspace",
        "this project",
        "my project",
        "repo",
        "repository",
        "codebase",
        "this file",
        "current file",
        "local file",
        "folder",
        "directory",
    }
    return _contains_any(text, local_terms)


def wants_attached_file(user_message: str) -> bool:
    return _contains_any(_normalise(user_message), ATTACHMENT_KEYWORDS)


def wants_mutating_tools(user_message: str) -> bool:
    return _contains_any(_normalise(user_message), MUTATING_KEYWORDS)


def format_policy_context(policy: QueryPolicy) -> str:
    return (
        "Context routing decision: "
        f"{policy.route}. {policy.reason} "
        "Use this routing decision when choosing whether to answer directly, use local files, uploaded files, or web sources."
    )


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())
