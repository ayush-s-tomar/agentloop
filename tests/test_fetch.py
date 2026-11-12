import pytest

from agent import llm, tools
from agent.graph import research_node


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("FETCH_PAGES", raising=False)


def run(monkeypatch, snippet, page):
    seen = {}
    monkeypatch.setattr(llm, "should_search", lambda q: True)
    monkeypatch.setattr(tools, "web_search", lambda q: [{"url": "u", "title": "t", "content": snippet, "score": 0.9}])
    monkeypatch.setattr(tools, "fetch_page", lambda u: page)
    monkeypatch.setattr(llm, "answer_from_context", lambda q, text: seen.setdefault("text", text) or "a")
    research_node({"plan": ["a new question here"], "completed_questions": [], "notes": [], "trace": []})
    return seen["text"]


def test_thin_snippet_triggers_fetch(monkeypatch):
    assert "FULLPAGE" in run(monkeypatch, "short", "FULLPAGE text")


def test_long_snippet_skips_fetch(monkeypatch):
    assert "FULLPAGE" not in run(monkeypatch, "x" * 500, "FULLPAGE text")


def test_failed_fetch_keeps_snippet(monkeypatch):
    assert "short" in run(monkeypatch, "short", "")


def test_fetch_disabled_by_env(monkeypatch):
    monkeypatch.setenv("FETCH_PAGES", "0")
    assert "FULLPAGE" not in run(monkeypatch, "short", "FULLPAGE text")


def test_fetch_page_fails_soft(monkeypatch):
    class Boom:
        def extract(self, urls):
            raise RuntimeError("x")
    monkeypatch.setattr(tools, "_get_client", lambda: Boom())
    assert tools.fetch_page("u") == ""