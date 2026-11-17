"""A/B eval for the fetch_page tool: FETCH_PAGES=0 vs FETCH_PAGES=1, judged per report.

Spends API quota. Use --cache so both arms replay identical search results:
    python evals/ab_fetch.py --n 3 --cache
Timings are not reported (cached search makes them meaningless).
"""
import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import tools  # noqa: E402
from evals.compare import DIMS, judge, total  # noqa: E402
from evals.run_eval import TOPICS  # noqa: E402


def summarise(rows):
    """Pair arms by (topic, repeat); only pairs where both arms were judged count."""
    pairs = {}
    for r in rows:
        if r.get("scores") and not r.get("error"):
            pairs.setdefault((r["topic"], r["repeat"]), {})[r["arm"]] = r
    ok = [p for p in pairs.values() if "on" in p and "off" in p]
    if not ok:
        return {"rows": len(rows), "pairs": 0}
    tot = lambda arm: [total(p[arm]["scores"]) for p in ok]  # noqa: E731
    on, off = tot("on"), tot("off")
    wins = sum(1 for a, b in zip(on, off) if a > b)
    ties = sum(1 for a, b in zip(on, off) if a == b)
    return {
        "rows": len(rows),
        "pairs": len(ok),
        "fetch_on_total_mean": round(statistics.mean(on), 2),
        "fetch_off_total_mean": round(statistics.mean(off), 2),
        "on_wins": wins,
        "ties": ties,
        "off_wins": len(ok) - wins - ties,
        "support_mean_on": round(statistics.mean(p["on"]["scores"]["support"] for p in ok), 2),
        "support_mean_off": round(statistics.mean(p["off"]["scores"]["support"] for p in ok), 2),
        "missing_figures_on": sum(p["on"].get("missing") or 0 for p in ok),
        "missing_figures_off": sum(p["off"].get("missing") or 0 for p in ok),
        "fetch_attempts_on": sum(p["on"].get("fetch_attempts", 0) for p in ok),
        "fetch_successes_on": sum(p["on"].get("fetch_successes", 0) for p in ok),
    }


def run_arm(topic, fetch_on):
    from agent.graph import run
    from memory import store

    store.init_db()
    os.environ["FETCH_PAGES"] = "1" if fetch_on else "0"
    stats = {"attempts": 0, "ok": 0}
    real = tools.fetch_page

    def counting(url, *a, **k):
        stats["attempts"] += 1
        page = real(url, *a, **k)
        if page:
            stats["ok"] += 1
        return page

    tools.fetch_page = counting
    final, failed = {}, 0
    try:
        for node, snap in run(topic, owner="ab-" + uuid.uuid4().hex):
            if node == "research":
                tr = snap.get("trace") or []
                if tr and str(tr[-1].get("message", "")).startswith("Researching (failed)"):
                    failed += 1
            elif node == "synthesize":
                final = snap
    finally:
        tools.fetch_page = real
        os.environ.pop("FETCH_PAGES", None)
    return final.get("final_report", ""), final.get("grounding") or {}, failed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--pause", type=float, default=10.0)
    ap.add_argument("--cache", action="store_true", help="replay identical search results from disk")
    args = ap.parse_args()
    if args.cache:
        os.environ["SEARCH_CACHE_DIR"] = str(Path(__file__).resolve().parent / ".search_cache")

    rows = []
    for topic in TOPICS[: max(1, min(args.n, len(TOPICS)))]:
        for rep in range(1, max(1, args.repeats) + 1):
            order = (False, True) if rep % 2 else (True, False)  # alternate to spread cache-order effects
            for fetch_on in order:
                arm = "on" if fetch_on else "off"
                print("[%s] run %d fetch=%s" % (topic, rep, arm), flush=True)
                row = {"topic": topic, "repeat": rep, "arm": arm}
                try:
                    report, ground, failed, st = run_arm(topic, fetch_on)
                    row["fetch_attempts"], row["fetch_successes"] = st["attempts"], st["ok"]
                    row["missing"] = ground.get("missing_count")
                    if failed or not report:
                        row["error"] = "invalid run (search failure or empty report)"
                    else:
                        row["scores"] = judge(topic, report)
                except Exception as exc:
                    row["error"] = str(exc)[:200]
                print("    ", {k: v for k, v in row.items() if k not in ("topic",)}, flush=True)
                rows.append(row)
                time.sleep(args.pause)

    summary = summarise(rows)
    out = Path(__file__).resolve().parent / "ab_fetch_results.json"
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()