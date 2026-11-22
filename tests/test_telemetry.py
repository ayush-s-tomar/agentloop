import pytest

from agent import telemetry
from agent.state import AgentState


@pytest.fixture(autouse=True)
def clean():
    telemetry._runs.clear()
    telemetry._errors["count"] = 0
    yield
    telemetry._runs.clear()


def research(msg, **kw):
    return dict({"trace": [{"node": "research", "message": msg}], "input_tokens": 10, "output_tokens": 5}, **kw)


def test_state_declares_token_keys():
    assert "input_tokens" in AgentState.__annotations__ and "output_tokens" in AgentState.__annotations__


def test_observe_counts_steps_and_tokens():
    acc = telemetry.new_acc()
    telemetry.observe(acc, "research", research("Researching (web): q"))
    telemetry.observe(acc, "research", research("Researching (failed): q"))
    telemetry.observe(acc, "research", research("Researching (skipped duplicate): q"))
    telemetry.observe(acc, "synthesize", {"grounding": {"missing_count": 2}, "all_sources": ["a", "b"]})
    assert (acc["questions"], acc["web"], acc["failed"], acc["skipped"]) == (2, 1, 1, 1)
    assert acc["in"] == 30 and acc["out"] == 15 and acc["missing"] == 2 and acc["sources"] == 2


def test_observe_never_raises():
    telemetry.observe(telemetry.new_acc(), "research", {"trace": None, "input_tokens": "x"})


def test_summary_aggregates(monkeypatch):
    monkeypatch.delenv("TAVILY_DEPTH", raising=False)
    for secs, web in ((10, 4), (30, 6)):
        acc = telemetry.new_acc()
        acc["web"] = web
        telemetry.record_run(acc, secs)
    s = telemetry.summary()
    assert s["runs"] == 2 and s["median_seconds"] == 20.0 and s["tavily_credits_upper_bound"] == 20


def test_basic_depth_halves_credit_bound(monkeypatch):
    monkeypatch.setenv("TAVILY_DEPTH", "basic")
    acc = telemetry.new_acc()
    acc["web"] = 5
    telemetry.record_run(acc, 1)
    assert telemetry.summary()["tavily_credits_upper_bound"] == 5


def test_empty_summary_and_endpoint():
    from fastapi.testclient import TestClient
    import main

    telemetry.record_error()
    body = TestClient(main.app).get("/api/stats").json()
    assert body == {"runs": 0, "errors": 1}