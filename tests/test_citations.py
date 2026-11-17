import pytest

from agent import llm
from agent.grounding import check_report, extract_numbers
from agent.graph import synthesize_node


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)


def test_citation_markers_are_not_figures():
    assert extract_numbers("Fines are capped [12] at 35 million [3][14].") == ["35"]


def test_report_with_citations_has_no_false_missing_figures():
    notes = [{"question": "q", "answer": "The cap is 35 million euros."}]
    r = check_report("The cap is 35 million euros [11].", notes, "t")
    assert r["missing_count"] == 0


def test_synthesis_receives_numbered_notes(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "synthesise", lambda topic, notes: seen.setdefault("n", notes) and "Report")
    state = {"topic": "t", "trace": [], "notes": [
        {"question": "q1", "answer": "a1", "sources": ["u1"]},
        {"question": "q2", "answer": "a2", "sources": []},
    ]}
    synthesize_node(state)
    assert seen["n"][0].startswith("[1] Q: q1") and seen["n"][1].startswith("[2] Q: q2")


def test_synthesis_prompt_asks_for_citations(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "_call", lambda messages, model, max_tokens, temperature=0.3: captured.setdefault("m", messages) and "ok")
    llm.synthesise("t", ["[1] Q: q"])
    assert "cite the supporting note number" in captured["m"][0]["content"]