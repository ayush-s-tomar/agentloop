"""Small evaluation harness for AgentLoop.

Runs the real agent (Groq and Tavily calls, so it spends quota) on a fixed
topic list and records completion, time, questions, sources, reflect loops and
how many report figures appear in the research notes.

    python evals/run_eval.py --n 5
    python evals/run_eval.py --n 10 --pause 8
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOPICS = [
    "How the EU AI Act affects startups",
    "Current state of solid-state EV batteries",
    "Impact of AI agents on software engineering jobs",
    "State of nuclear fusion energy research",
    "How GLP-1 drugs are changing the food industry",
    "Open-source large language models versus proprietary ones",
    "Challenges of vertical farming at commercial scale",
    "Progress of quantum error correction",
    "Remote work and commercial real estate demand",
    "Rise of small modular nuclear reactors",
]


def run_one(topic):
    from agent.graph import run

    start = time.time()
    final, researched, loops, error = {}, 0, 0, None
    try:
        for node, snap in run(topic):
            if node == "research":
                researched += 1
            elif node == "reflect" and snap.get("decision") == "more_research":
                loops += 1
            elif node == "synthesize":
                final = snap
    except Exception as exc:  # keep going so one failure does not end the eval
        error = str(exc)[:200]
    g = final.get("grounding") or {}
    return {
        "topic": topic,
        "completed": bool(final.get("final_report")) and error is None,
        "seconds": round(time.time() - start, 1),
        "questions": researched,
        "sources": len(final.get("all_sources") or []),
        "reflect_loops": loops,
        "figures_checked": g.get("checked", 0),
        "figures_missing": g.get("missing_count", 0),
        "words": len((final.get("final_report") or "").split()),
        "error": error,
    }


def summarise(rows):
    done = [r for r in rows if r["completed"]]
    checked = sum(r["figures_checked"] for r in done)
    missing = sum(r["figures_missing"] for r in done)
    return {
        "runs": len(rows),
        "completed": len(done),
        "median_seconds": statistics.median(r["seconds"] for r in done) if done else None,
        "mean_questions": round(statistics.mean(r["questions"] for r in done), 1) if done else None,
        "mean_sources": round(statistics.mean(r["sources"] for r in done), 1) if done else None,
        "runs_with_reflect_loop": sum(1 for r in done if r["reflect_loops"] > 0),
        "figures_checked": checked,
        "figures_missing": missing,
        "figures_in_notes_pct": round(100 * (1 - missing / checked), 1) if checked else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="how many topics to run (max 10)")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between runs")
    args = ap.parse_args()

    rows = []
    for i, topic in enumerate(TOPICS[: max(1, min(args.n, len(TOPICS)))], 1):
        print("[%d] %s" % (i, topic), flush=True)
        row = run_one(topic)
        print("    completed=%s %.1fs questions=%d sources=%d figures=%d missing=%d %s" % (
            row["completed"], row["seconds"], row["questions"], row["sources"],
            row["figures_checked"], row["figures_missing"], row["error"] or ""), flush=True)
        rows.append(row)
        time.sleep(args.pause)

    summary = summarise(rows)
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nWrote", out)


if __name__ == "__main__":
    main()