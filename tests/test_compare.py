import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "evals"))

import compare  # noqa: E402


def test_parse_judge_valid_and_fenced():
    assert compare.parse_judge('{"relevance": 4, "completeness": 3, "support": 5}') == {
        "relevance": 4, "completeness": 3, "support": 5}
    assert compare.parse_judge('```json\n{"relevance": 4, "completeness": 3, "support": 5}\n```')["support"] == 5


def test_parse_judge_clips_range():
    r = compare.parse_judge('{"relevance": 9, "completeness": 0, "support": 3}')
    assert r == {"relevance": 5, "completeness": 1, "support": 3}


def test_parse_judge_rejects_garbage():
    assert compare.parse_judge("not json") is None
    assert compare.parse_judge('{"relevance": 4}') is None
    assert compare.parse_judge(None) is None


def row(a, b):
    mk = lambda n: {"relevance": n, "completeness": n, "support": n}  # noqa: E731
    return {"agent_scores": mk(a), "base_scores": mk(b), "agent_seconds": 50, "base_seconds": 10}


def test_summarise_counts_wins_ties_losses():
    s = compare.summarise([row(5, 3), row(3, 3), row(2, 4), {"error": "x"}])
    assert s["judged"] == 3
    assert (s["agent_wins"], s["ties"], s["baseline_wins"]) == (1, 1, 1)


def test_summarise_handles_no_valid_rows():
    assert compare.summarise([{"error": "x"}])["judged"] == 0