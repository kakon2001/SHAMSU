from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any

SEARCH_URL = "https://duckduckgo.com/html/"
USER_AGENT = "SHAMSU/1.0 local-coding-agent"


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo-lite"


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._in_link = False
        self._in_snippet = False
        self._current_url = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_snippet_index: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._in_link = True
            self._current_url = _clean_duckduckgo_url(attrs_dict.get("href", ""))
            self._title_parts = []
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []
            self._pending_snippet_index = len(self.results) - 1 if self.results else None

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            title = _squash(" ".join(self._title_parts))
            if title and self._current_url:
                self.results.append(WebSearchResult(title=title, url=self._current_url, snippet=""))
            self._in_link = False
            self._current_url = ""
            self._title_parts = []
        elif tag in {"a", "div"} and self._in_snippet:
            snippet = _squash(" ".join(self._snippet_parts))
            if self._pending_snippet_index is not None and 0 <= self._pending_snippet_index < len(self.results):
                self.results[self._pending_snippet_index].snippet = snippet
            self._in_snippet = False
            self._snippet_parts = []
            self._pending_snippet_index = None


def search_web(query: str, limit: int = 5, timeout: float = 8.0) -> dict[str, Any]:
    query = _squash(query)
    limit = max(1, min(int(limit or 5), 10))
    if not query:
        return {"ok": False, "query": query, "provider": "duckduckgo-lite", "results": [], "error": "Query is required."}

    params = urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        f"{SEARCH_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {
            "ok": False,
            "query": query,
            "provider": "duckduckgo-lite",
            "results": [],
            "error": f"Web search unavailable: {exc}",
            "notes": ["Check internet access or use uploaded/workspace files for offline context."],
        }

    results = parse_duckduckgo_html(body, limit=limit)
    return {
        "ok": True,
        "query": query,
        "provider": "duckduckgo-lite",
        "results": [asdict(item) for item in results],
        "notes": ["Use source URLs to verify current facts before relying on them."],
    }


def parse_duckduckgo_html(body: str, limit: int = 5) -> list[WebSearchResult]:
    parser = DuckDuckGoHTMLParser()
    parser.feed(body)
    deduped: list[WebSearchResult] = []
    seen: set[str] = set()
    for result in parser.results:
        if result.url in seen:
            continue
        seen.add(result.url)
        deduped.append(result)
        if len(deduped) >= limit:
            break
    return deduped


def format_search_results(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"Error: {payload.get('error') or 'web search failed'}"
    results = payload.get("results") or []
    if not results:
        return "No web search results found."
    lines = [f"Web search results for: {payload.get('query')}"]
    for index, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        lines.append(f"{index}. {title}\n   {url}\n   {snippet}".rstrip())
    return "\n".join(lines)


def _clean_duckduckgo_url(url: str) -> str:
    url = html.unescape(url or "")
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()
