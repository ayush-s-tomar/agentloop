"""
Tools the agent can call.

This is real function-calling, not a hardcoded pipeline step: the LLM is
given the `web_search` tool's schema and decides on its own, per
sub-question, whether and how to call it (see agent/graph.py::research_node).
The agent never sees raw search results without going through this layer,
which is also where you'd add input validation, retries, or fallbacks to a
second provider if you took this to production.
"""

import os
from typing import List, Dict

from tavily import TavilyClient

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
        _client = TavilyClient(api_key=api_key)
    return _client


def _search_uncached(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Run a live web search and return [{title, url, content}, ...]."""
    response = _get_client().search(
        query=query,
        max_results=max_results,
        search_depth=os.environ.get("TAVILY_DEPTH", "advanced"),
    )
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:1500],  # keep prompt sizes sane
                "score": item.get("score"),
            }
        )
    return results


import time as _time
import json as _json
import hashlib as _hashlib
from pathlib import Path as _Path


def _disk_path(query, max_results):
    """Cache file for this query, or None when SEARCH_CACHE_DIR is unset (the default)."""
    d = os.environ.get("SEARCH_CACHE_DIR")
    if not d:
        return None
    key = "%s|%s|%s" % (" ".join(query.lower().split()), max_results, os.environ.get("TAVILY_DEPTH", "advanced"))
    return _Path(d) / (_hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def _search_with_disk_cache(query, max_results):
    path = _disk_path(query, max_results)
    if path is not None and path.exists():
        try:
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass  # unreadable cache file: fall through to a live search
    results = _search_uncached(query, max_results)
    if path is not None and results:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(results), encoding="utf-8")
        except Exception:
            pass
    return results

_CACHE: dict = {}
_CACHE_TTL_S = 3600
_CACHE_MAX = 64


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Cached wrapper: identical queries within an hour reuse the earlier results (saves Tavily credits)."""
    key = (" ".join(query.lower().split()), max_results)
    hit = _CACHE.get(key)
    now = _time.time()
    if hit and now - hit[0] < _CACHE_TTL_S:
        return hit[1]
    results = _search_with_disk_cache(query, max_results)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (now, results)
    return results


def fetch_page(url: str, max_chars: int = 3000) -> str:
    """Second tool: read the full text of one page. Returns "" on any failure."""
    try:
        resp = _get_client().extract(urls=[url])
        items = resp.get("results") or []
        if not items:
            return ""
        return (items[0].get("raw_content") or "")[:max_chars]
    except Exception:
        return ""


# OpenAI-compatible tool schema — Groq's chat completions API speaks this format.
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for up-to-date information on a specific question. "
            "Use this whenever you need facts, news, or data you are not certain about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, e.g. 'EU AI Act enforcement timeline 2026'.",
                },
            },
            "required": ["query"],
        },
    },
}
