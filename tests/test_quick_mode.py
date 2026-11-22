import inspect
import pathlib

import pytest

from agent import graph, llm
from agent.graph import planner_node, reflect_node
from agent.state import AgentState

HTML = (pathlib.Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD", "MAX_SUBQUESTIONS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(llm, "plan", lambda topic, prior: ["a", "b", "c", "d"])


def test_state_declares_mode():
    assert "mode" in AgentState.__annotations__


def test_run_accepts_mode():
    assert "mode" in inspect.signature(graph.run).parameters


def test_quick_plans_two_questions():
    assert len(planner_node({"topic": "t", "mode": "quick", "trace": []})["plan"]) == 2


def test_deep_plans_four_questions():
    assert len(planner_node({"topic": "t", "mode": "deep", "trace": []})["plan"]) == 4


def test_quick_skips_reflection(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "reflect", boom)
    out = reflect_node({"topic": "t", "reflect_iterations": 0, "max_reflect_iterations": 0, "notes": [], "trace": []})
    assert out["decision"] == "done"


def test_ui_sends_mode():
    assert 'id="mode"' in HTML and 'mode:document.getElementById("mode").value' in HTML