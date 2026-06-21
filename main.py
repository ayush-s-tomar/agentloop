"""
AgentLoop — FastAPI backend
  GET  /           → serve frontend HTML
  GET  /health     → {"status": "ok"}
  POST /research   → SSE stream of node events + final report
"""
from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from time import time

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.memory import init_db
from agent.graph  import run as agent_run

# ─── logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ─── app ──────────────────────────────────────────────────────
app = FastAPI(title="AgentLoop", version="1.0.0")
init_db()

# ─── simple in-memory rate limiter (10 req / 60 s per IP) ─────
RATE_LIMIT   = 10
RATE_WINDOW  = 60
_rate_store: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(ip: str) -> None:
    now   = time()
    calls = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
    if len(calls) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
    calls.append(now)
    _rate_store[ip] = calls


# ─── models ───────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str


# ─── routes ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/research")
async def research(req: Request, body: ResearchRequest):
    check_rate_limit(req.client.host)
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if len(topic) > 300:
        raise HTTPException(status_code=400, detail="Topic too long (max 300 chars).")

    log.info("Research request: topic=%r ip=%s", topic, req.client.host)

    NODE_LABELS = {
        "recall":     "Checking memory …",
        "plan":       "Planning sub-questions …",
        "research":   "Researching …",
        "reflect":    "Reflecting on findings …",
        "synthesise": "Writing report …",
        "persist":    "Saving to memory …",
    }

    def event_stream():
        try:
            final_report = ""
            for node_name, snapshot in agent_run(topic):
                label = NODE_LABELS.get(node_name, node_name)
                payload: dict = {"node": node_name, "label": label}

                if node_name == "plan":
                    payload["sub_questions"] = snapshot.get("sub_questions", [])
                elif node_name == "reflect":
                    payload["reflection"] = snapshot.get("reflection", "")
                elif node_name == "synthesise":
                    final_report = snapshot.get("report", "")

                yield f"data: {json.dumps(payload)}\n\n"

            # send final report after all nodes complete
            yield f"data: {json.dumps({'node': '__done__', 'report': final_report})}\n\n"

        except Exception as exc:
            log.error("Stream error: %s", exc, exc_info=True)
            # Don't leak internal details (org ids, token counts, etc.)
            safe_msg = "An error occurred during research. Please try again in a moment."
            yield f"data: {json.dumps({'node': '__error__', 'message': safe_msg})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
