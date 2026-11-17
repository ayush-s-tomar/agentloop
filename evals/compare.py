"""Baseline vs AgentLoop comparison with an LLM judge.

Baseline = one Tavily search + one LLM call. Spends API quota:
    python evals/compare.py --n 3 --pause 15
"""
import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import llm  # noqa: E402
from agent.grounding import check_report  # noqa: E402
from evals.run_eval import TOPICS  # noqa: E402

DIMS = ("relevance", "completeness", "support")

JUDGE_SYSTEM = (
    "You are a strict research-report grader. Score the report on a 1-5 scale for: "
    "relevance (answers the topic), completeness (covers the main aspects), "
    "support (claims look specific and sourced rather than vague or invented; penalise "
    "unexplained numbers). Reply with raw JSON only: "
    '{"relevance": n, "completeness": n, "support": n}'
)


def parse_judge(raw):
    """Return {dim: int 1-5} or None if the judge reply is unusable."""
    try:
        data = json.loads(llm._strip_fences(raw or ""))
        out = {}
        for d in DIMS:
            out[d] = max(1, min(5, int(round(float(data[d])))))
        return out
    except Exception:
        return None


def judge(topic, report):
    raw = llm._call(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "Topic: %s\n\nReport:\n%s" % (topic, report[:6000])},
        ],
        model=llm.REASON_MODEL,
        max_tokens=800,
        temperature=0.0,
    )
    return parse_judge(raw)


def total(scores):
    return sum(scores[d] for d in DIMS) if scores else None


def summarise(rows):
    ok = [r for r in rows if r.get("agent_scores") and r.get("base_scores")]
    if not ok:
        return {"runs": len(rows), "judged": 0}
    mean = lambda key, d: round(statistics.mean(r[key][d] for r in ok), 2)  # noqa: E731
    wins = sum(1 for r in ok if total(r["agent_scores"]) > total(r["base_scores"]))
    ties = sum(1 for r in ok if total(r["agent_scores"]) == total(r["base_scores"]))
    return {
        "runs": len(rows),
        "judged": len(ok),
        "agent_mean": {d: mean("agent_scores", d) for d in DIMS},
        "baseline_mean": {d: mean("base_scores", d) for d in DIMS},
        "agent_wins": wins,
        "ties": ties,
        "baseline_wins": len(ok) - wins - ties,
        "median_seconds_agent": round(statistics.median(r["agent_seconds"] for r in ok), 1),
        "median_seconds_baseline": round(statistics.median(r["base_seconds"] for r in ok), 1),
    }


def baseline_report(topic):
    from agent import tools

    results = tools.web_search(topic)
    text = "\n\n".join("Title: %s\n%s" % (r.get("title", ""), r.get("content", "")) for r in results)
    report = llm._call(
        messages=[
            {"role": "system", "content": (
                "Write a structured Markdown research report (Overview, Key Findings, Analysis, "
                "Conclusion) using ONLY the provided search results. Do not name any number not "
                "in the results. Under about 900 words.")},
            {"role": "user", "content": "Topic: %s\n\nSearch results:\n%s" % (topic, text)},
        ],
        model=llm.REASON_MODEL,
        max_tokens=3200,
        temperature=0.4,
    )
    notes = [{"question": topic, "answer": text}]
    return report, notes


def agent_report(topic):
    from agent.graph import run
    from memory import store

    store.init_db()
    final, failed = {}, 0
    for node, snap in run(topic, owner="cmp-" + uuid.uuid4().hex):
        if node == "research":
            tr = snap.get("trace") or []
            if tr and str(tr[-1].get("message", "")).startswith("Researching (failed)"):
                failed += 1
        elif node == "synthesize":
            final = snap
    return final.get("final_report", ""), final.get("grounding") or {}, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pause", type=float, default=15.0)
    args = ap.parse_args()

    rows = []
    for i, topic in enumerate(TOPICS[: max(1, min(args.n, len(TOPICS)))], 1):
        print("[%d] %s" % (i, topic), flush=True)
        row = {"topic": topic}
        try:
            t0 = time.time()
            a_report, a_ground, failed = agent_report(topic)
            row["agent_seconds"] = round(time.time() - t0, 1)
            row["agent_search_failures"] = failed
            row["agent_missing_figures"] = a_ground.get("missing_count")

            t0 = time.time()
            b_report, b_notes = baseline_report(topic)
            row["base_seconds"] = round(time.time() - t0, 1)
            b_ground = check_report(b_report, b_notes, topic)
            row["base_missing_figures"] = b_ground.get("missing_count")

            if failed or not a_report or not b_report:
                row["error"] = "invalid run (search failure or empty report)"
            else:
                row["agent_scores"] = judge(topic, a_report)
                row["base_scores"] = judge(topic, b_report)
        except Exception as exc:
            row["error"] = str(exc)[:200]
        print("    ", {k: v for k, v in row.items() if k != "topic"}, flush=True)
        rows.append(row)
        time.sleep(args.pause)

    summary = summarise(rows)
    out = Path(__file__).resolve().parent / "compare_results.json"
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()