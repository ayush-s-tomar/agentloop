"""
FastAPI surface for the agent.

  POST /api/research            run the agent end-to-end, return final result
  GET  /api/research/stream      same run, but streamed step-by-step over
                                  Server-Sent Events — this is what the demo
                                  UI uses so you can literally watch the
                                  agent plan -> search -> reflect -> write
  GET  /api/history              list past runs (long-term memory)
  GET  /api/history/{session_id} fetch one past run in full
  GET  /                         the demo UI
"""

import json
import os
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import graph
from memory import store

store.init_db()

app = FastAPI(title="AgentLoop", description="Tool-using research agent with memory")


class ResearchRequest(BaseModel):
    topic: str


def _initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "session_id": store.new_session_id(),
        "trace": [],
        "max_reflect_iterations": int(os.environ.get("MAX_REFLECT_ITERATIONS", "2")),
    }


@app.post("/api/research")
def run_research(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    final_state = graph.invoke(_initial_state(req.topic))
    return {
        "session_id": final_state["session_id"],
        "topic": final_state["topic"],
        "report": final_state.get("final_report", ""),
        "notes": final_state.get("notes", []),
        "trace": final_state.get("trace", []),
        "recalled_context": final_state.get("recalled_context"),
    }


@app.get("/api/research/stream")
async def stream_research(topic: str) -> StreamingResponse:
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    async def event_stream() -> AsyncIterator[str]:
        seen_trace_len = 0
        final_state = {}
        try:
            for step in graph.stream(_initial_state(topic), stream_mode="values"):
                final_state = step
                trace = step.get("trace", [])
                for event in trace[seen_trace_len:]:
                    yield f"data: {json.dumps({'type': 'trace', **event})}\n\n"
                seen_trace_len = len(trace)

            yield f"data: {json.dumps({'type': 'done', 'session_id': final_state.get('session_id'), 'report': final_state.get('final_report', ''), 'notes': final_state.get('notes', [])})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
