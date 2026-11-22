"""In-memory run telemetry: numbers only (no topics, no user ids). Resets on restart."""
import os
import statistics
import threading
from collections import deque

_lock = threading.Lock()
_runs = deque(maxlen=200)
_errors = {"count": 0}


def new_acc():
    return {"in": 0, "out": 0, "questions": 0, "web": 0, "failed": 0, "skipped": 0, "missing": 0, "sources": 0}


def observe(acc, node, snap):
    """Fold one node snapshot into the accumulator. Never raises."""
    try:
        snap = snap or {}
        acc["in"] += int(snap.get("input_tokens") or 0)
        acc["out"] += int(snap.get("output_tokens") or 0)
        if node == "research":
            msg = str(((snap.get("trace") or [{}])[-1]).get("message", ""))
            if "skipped duplicate" in msg:
                acc["skipped"] += 1
            else:
                acc["questions"] += 1
                if msg.startswith("Researching (web)"):
                    acc["web"] += 1
                elif msg.startswith("Researching (failed)"):
                    acc["failed"] += 1
        elif node == "synthesize":
            acc["missing"] = int((snap.get("grounding") or {}).get("missing_count") or 0)
            acc["sources"] = len(snap.get("all_sources") or [])
    except Exception:
        pass


def record_run(acc, seconds, fallbacks=0):
    with _lock:
        _runs.append(dict(acc, seconds=round(float(seconds), 1), fallbacks=max(0, int(fallbacks))))


def record_error():
    with _lock:
        _errors["count"] += 1


def summary():
    with _lock:
        runs = list(_runs)
        errors = _errors["count"]
    if not runs:
        return {"runs": 0, "errors": errors}
    secs = sorted(r["seconds"] for r in runs)
    per_search = 1 if os.environ.get("TAVILY_DEPTH", "advanced") == "basic" else 2
    mean = lambda k: round(statistics.mean(r[k] for r in runs), 1)  # noqa: E731
    return {
        "runs": len(runs),
        "errors": errors,
        "median_seconds": round(statistics.median(secs), 1),
        "p90_seconds": secs[min(len(secs) - 1, int(len(secs) * 0.9))],
        "mean_questions": mean("questions"),
        "mean_web_searches": mean("web"),
        "mean_input_tokens": int(statistics.mean(r["in"] for r in runs)),
        "mean_output_tokens": int(statistics.mean(r["out"] for r in runs)),
        "runs_with_failed_steps": sum(1 for r in runs if r["failed"]),
        "skipped_duplicates": sum(r["skipped"] for r in runs),
        "model_fallbacks": sum(r["fallbacks"] for r in runs),
        "unsupported_figures_total": sum(r["missing"] for r in runs),
        "tavily_credits_upper_bound": sum(r["web"] for r in runs) * per_search,
        "window": "last %d runs, this instance only" % _runs.maxlen,
    }