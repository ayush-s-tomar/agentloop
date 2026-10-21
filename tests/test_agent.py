import pytest

from agent import llm
from agent.graph import reflect_node, route_after_reflect, route_after_research


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)


def stub_llm(monkeypatch, reply):
    calls = []

    def fake_call(messages, model, max_tokens, temperature=0.3):
        calls.append(messages)
        return reply

    monkeypatch.setattr(llm, "_call", fake_call)
    return calls


def test_route_after_research():
    assert route_after_research({"plan": ["q1"]}) == "research"
    assert route_after_research({"plan": []}) == "reflect"
    assert route_after_research({}) == "reflect"


def test_route_after_reflect():
    assert route_after_reflect({"decision": "more_research"}) == "research"
    assert route_after_reflect({"decision": "done"}) == "synthesize"
    assert route_after_reflect({}) == "synthesize"


def test_reflect_node_stops_at_iteration_cap(monkeypatch):
    calls = stub_llm(monkeypatch, "{}")
    state = {"topic": "t", "reflect_iterations": 2, "max_reflect_iterations": 2,
             "notes": [], "trace": []}
    assert reflect_node(state)["decision"] == "done"
    assert calls == []


def test_strip_fences():
    assert llm._strip_fences("```json\n[\"a\"]\n```") == "[\"a\"]"


def test_should_search_keyword_skips_llm(monkeypatch):
    calls = stub_llm(monkeypatch, "NO")
    assert llm.should_search("What is the latest Python version?") is True
    assert calls == []


@pytest.mark.parametrize("reply,expected", [("NO", False), ("no", False), ("YES", True), ("Maybe", True)])
def test_should_search_uses_llm_answer(monkeypatch, reply, expected):
    stub_llm(monkeypatch, reply)
    assert llm.should_search("Explain how photosynthesis works") is expected


def test_plan_parses_fenced_json(monkeypatch):
    stub_llm(monkeypatch, "```json\n[\"q1\", \"q2\"]\n```")
    assert llm.plan("topic", []) == ["q1", "q2"]


def test_reflect_caps_followups_at_two(monkeypatch):
    stub_llm(monkeypatch, '{"sufficient": false, "additional_questions": ["a", "b", "c"]}')
    assert llm.reflect("t", ["n"]) == (True, ["a", "b"])


def test_reflect_sufficient(monkeypatch):
    stub_llm(monkeypatch, '{"sufficient": true, "additional_questions": []}')
    assert llm.reflect("t", ["n"]) == (False, [])