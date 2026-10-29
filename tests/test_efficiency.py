import pytest

from agent import llm, tools
from agent.graph import _too_similar, research_node


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)


def test_too_similar_detects_near_duplicates():
    done = ["What impact does the EU AI Act risk classification have on the product roadmap?"]
    assert _too_similar("What impact does the EU AI Act risk classification have on the product roadmap of startups?", done)
    assert not _too_similar("What are the costs of conformity assessments?", done)


def test_research_node_skips_duplicate_question(monkeypatch):
    monkeypatch.setattr(llm, "should_search", lambda q: False)
    monkeypatch.setattr(llm, "answer_from_knowledge", lambda q: "answer")
    q = "How does the risk classification system affect startup roadmaps"
    state = {"plan": [q], "completed_questions": [q], "notes": [], "trace": []}
    out = research_node(state)
    assert "notes" not in out
    assert out["plan"] == []


def test_research_node_keeps_distinct_question(monkeypatch):
    monkeypatch.setattr(llm, "should_search", lambda q: False)
    monkeypatch.setattr(llm, "answer_from_knowledge", lambda q: "answer")
    state = {"plan": ["What are the compliance costs?"], "completed_questions": ["Who founded the company?"], "notes": [], "trace": []}
    out = research_node(state)
    assert len(out["notes"]) == 1


def test_search_cache_avoids_second_call(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_search_uncached", lambda q, n=5: calls.append(q) or [{"url": "u"}])
    tools._CACHE.clear()
    tools.web_search("EU AI Act")
    tools.web_search("  eu ai act ")
    assert len(calls) == 1
    tools._CACHE.clear()