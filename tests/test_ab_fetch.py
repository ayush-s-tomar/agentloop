import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "evals"))

import ab_fetch  # noqa: E402


def row(topic, arm, n, **kw):
    r = {"topic": topic, "repeat": 1, "arm": arm,
         "scores": {"relevance": n, "completeness": n, "support": n}, "missing": 0}
    r.update(kw)
    return r


def test_pairs_arms_and_counts_wins():
    rows = [row("a", "on", 5, fetch_attempts=2, fetch_successes=1), row("a", "off", 3),
            row("b", "on", 3), row("b", "off", 3)]
    s = ab_fetch.summarise(rows)
    assert s["pairs"] == 2 and (s["on_wins"], s["ties"], s["off_wins"]) == (1, 1, 0)
    assert s["fetch_attempts_on"] == 2 and s["fetch_successes_on"] == 1


def test_unpaired_or_errored_rows_are_excluded():
    rows = [row("a", "on", 5), {"topic": "a", "repeat": 1, "arm": "off", "error": "x"}]
    assert ab_fetch.summarise(rows)["pairs"] == 0


def test_compare_runs_use_isolated_owner():
    src = (pathlib.Path(__file__).resolve().parent.parent / "evals" / "compare.py").read_text(encoding="utf-8")
    assert 'owner="cmp-"' in src