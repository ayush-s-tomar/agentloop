import pytest

from agent import llm
from agent.graph import synthesize_node
from agent.state import AgentState
import memory.store as store

A = "aaaaaaaa-1111"
B = "bbbbbbbb-2222"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.init_db()


def seed():
    store.save_session("s1", "solid state batteries", [], [], "report a", owner=A)
    store.save_session("s2", "eu ai act startups", [], [], "report b", owner=B)


def test_list_is_scoped_to_owner():
    seed()
    assert [s["id"] for s in store.list_sessions(owner=A)] == ["s1"]
    assert [s["id"] for s in store.list_sessions(owner=B)] == ["s2"]
    assert store.list_sessions(owner="") == []


def test_get_session_checks_owner():
    seed()
    assert store.get_session("s2", owner=A) is None
    assert store.get_session("s2", owner=B)["report"] == "report b"


def test_delete_all_only_removes_own_sessions():
    seed()
    store.delete_all_sessions(owner=A)
    assert store.list_sessions(owner=A) == []
    assert [s["id"] for s in store.list_sessions(owner=B)] == ["s2"]


def test_recall_does_not_cross_owners():
    seed()
    assert store.find_similar_session("eu ai act startups", owner=A) is None
    assert store.find_similar_session("eu ai act startups", owner=B)["id"] == "s2"


def test_api_scopes_sessions_by_header():
    from fastapi.testclient import TestClient
    import main

    seed()
    c = TestClient(main.app)
    ha, hb = {"X-Client-Id": A}, {"X-Client-Id": B}
    assert [s["id"] for s in c.get("/api/sessions", headers=ha).json()] == ["s1"]
    assert c.get("/api/sessions").json() == []
    assert c.get("/api/sessions/s2", headers=ha).json() == {"error": "not found"}
    c.delete("/api/sessions", headers=ha)
    assert [s["id"] for s in c.get("/api/sessions", headers=hb).json()] == ["s2"]


def test_state_has_owner_and_trail():
    assert "owner" in AgentState.__annotations__
    assert "trail" in AgentState.__annotations__


def test_synthesize_returns_per_question_trail(monkeypatch):
    monkeypatch.setattr(llm, "synthesise", lambda topic, notes: "Report")
    state = {
        "topic": "t",
        "notes": [
            {"question": "q1", "answer": "a", "sources": ["u1"]},
            {"question": "q2", "answer": "b", "sources": []},
        ],
        "trace": [],
    }
    out = synthesize_node(state)
    assert out["trail"] == [
        {"question": "q1", "sources": ["u1"]},
        {"question": "q2", "sources": []},
    ]