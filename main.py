"""
FastAPI surface for the agent.

  POST /api/research/start       kick off a run in the background, return job_id
  GET  /api/research/status/{id} poll for live trace + completion (frontend polls every 2s)
  POST /api/research             run the agent end-to-end synchronously (fallback)
  GET  /api/history              list past runs (long-term memory)
  GET  /api/history/{session_id} fetch one past run in full
  GET  /                         the demo UI

Background-task + polling architecture avoids SSE timeout issues on free-tier
hosting (Render, Railway, etc.) where long-lived streaming connections get cut.
"""

import json
import os
import threading
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import graph
from memory import store

store.init_db()

app = FastAPI(title="AgentLoop", description="Tool-using research agent with memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store — good enough for a demo; resets on restart
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class ResearchRequest(BaseModel):
    topic: str


def _initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "session_id": store.new_session_id(),
        "trace": [],
        "max_reflect_iterations": int(os.environ.get("MAX_REFLECT_ITERATIONS", "2")),
    }


def _run_agent(job_id: str, topic: str):
    """Runs in a background thread. Updates _jobs[job_id] as nodes complete."""
    try:
        seen_trace_len = 0
        final_state = {}

        for step in graph.stream(_initial_state(topic), stream_mode="values"):
            final_state = step
            trace = step.get("trace", [])
            new_events = trace[seen_trace_len:]
            seen_trace_len = len(trace)

            with _jobs_lock:
                _jobs[job_id]["trace"].extend(new_events)
                # Update current node indicator
                if new_events:
                    _jobs[job_id]["current_node"] = new_events[-1].get("node", "")

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "report": final_state.get("final_report", ""),
                "notes": final_state.get("notes", []),
                "session_id": final_state.get("session_id"),
                "current_node": "done",
            })

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "error": str(exc),
                "current_node": "error",
            })


@app.post("/api/research/start")
def start_research(req: ResearchRequest):
    """Kick off a background research job. Returns job_id immediately."""
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    job_id = store.new_session_id()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "topic": req.topic,
            "trace": [],
            "current_node": "recall",
            "report": None,
            "notes": [],
            "error": None,
        }

    t = threading.Thread(target=_run_agent, args=(job_id, req.topic), daemon=True)
    t.start()

    return {"job_id": job_id}


@app.get("/api/research/status/{job_id}")
def job_status(job_id: str):
    """Poll this endpoint to get live trace updates and final report."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/api/research")
def run_research(req: ResearchRequest):
    """Synchronous fallback — runs the full agent and returns when done."""
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    final_state = graph.invoke(_initial_state(req.topic))
    return {
        "session_id": final_state["session_id"],
        "topic": final_state["topic"],
        "report": final_state.get("final_report", ""),
        "notes": final_state.get("notes", []),
        "trace": final_state.get("trace", []),
    }


@app.get("/api/history")
def history(limit: int = 20):
    return store.list_sessions(limit=limit)


@app.get("/api/history/{session_id}")
def history_detail(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return session


app.mount("/", StaticFiles(directory="static", html=True), name="static")
