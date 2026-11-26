import pathlib

import pytest
from fastapi.testclient import TestClient

import main
import memory.store as store
from memory import feedback

HTML = (pathlib.Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
A, B = "aaaaaaaa-1111", "bbbbbbbb-2222"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.init_db()
    store.save_session("s1", "topic a", [], [], "report a", owner=A)
    store.save_session("s2", "topic b", [], [], "report b", owner=B)


def test_save_replaces_vote_and_is_owner_scoped():
    feedback.save("s1", A, 1)
    assert feedback.get("s1", A) == 1
    feedback.save("s1", A, -1)
    assert feedback.get("s1", A) == -1
    assert feedback.get("s1", B) == 0


def test_invalid_rating_rejected():
    with pytest.raises(ValueError):
        feedback.save("s1", A, 5)


def test_summary_counts():
    feedback.save("s1", A, 1)
    feedback.save("s2", B, -1)
    assert feedback.summary() == {"up": 1, "down": 1}


def test_endpoints():
    c = TestClient(main.app)
    ha, hb = {"X-Client-Id": A}, {"X-Client-Id": B}
    assert c.post("/api/feedback", json={"session_id": "s1", "rating": 1}, headers=ha).json() == {"ok": True}
    assert c.get("/api/feedback/s1", headers=ha).json() == {"rating": 1}
    assert c.get("/api/feedback/s1", headers=hb).json() == {"rating": 0}
    assert c.post("/api/feedback", json={"session_id": "s1", "rating": 1}, headers=hb).status_code == 404
    assert c.post("/api/feedback", json={"session_id": "s1", "rating": 3}, headers=ha).status_code == 400
    assert c.post("/api/feedback", json={"session_id": "s1", "rating": 1}).status_code == 400
    assert c.get("/api/feedback_summary").json() == {"up": 1, "down": 0}


def test_ui_is_wired():
    assert 'id="fbUp"' in HTML and "function sendFeedback" in HTML and "refreshFeedback();" in HTML