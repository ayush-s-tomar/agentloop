# AgentLoop

**A multi-step research agent with tool-use and memory.**

Most AI projects are input → output. AgentLoop is different — it plans, searches the live web, reflects on its own gaps, loops back if needed, then writes a structured report. Built to demonstrate agentic AI, not just LLM wrapping.

🔗 **[Live Demo](https://agentloop.onrender.com)**

---

## What it does

Give it any topic. The agent runs a 6-step pipeline autonomously:

| Step | What happens |
|------|-------------|
| **Recall** | Checks SQLite long-term memory for related past research |
| **Plan** | LLM breaks the topic into specific sub-questions |
| **Research** | For each sub-question, LLM *decides* whether to call `web_search` (Tavily), reads results, writes a cited answer |
| **Reflect** | Re-reads its own notes, identifies gaps, loops back to research if needed |
| **Synthesize** | Writes a structured markdown report from everything gathered |
| **Persist** | Saves the run to long-term memory for future recall |

```
START → recall → planner → research ←─────────┐
                               │               │ (loop while sub-questions remain)
                               ▼               │
                            reflect ───────────┘ (loop back if gaps found)
                               │
                               ▼
                          synthesize → persist → END
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph (StateGraph with conditional edges) |
| LLM + tool-calling | Groq (`llama-3.3-70b-versatile`) |
| Web search tool | Tavily API |
| Backend | FastAPI |
| Long-term memory | SQLite |
| Frontend | Vanilla JS + SSE live trace |
| Deploy | Render |

---

## Project structure

```
agentloop/
├── main.py              FastAPI app — REST + background polling endpoints
├── agent/
│   ├── state.py         AgentState schema shared across all graph nodes
│   ├── graph.py         LangGraph StateGraph: nodes + conditional routing
│   ├── llm.py           LLM wrapper: plain completions + tool-calling loop
│   └── tools.py         web_search tool (Tavily) + OpenAI-compatible schema
├── memory/
│   └── store.py         SQLite long-term memory (save + recall past sessions)
├── static/
│   └── index.html       Single-page demo UI with live pipeline stepper
├── Dockerfile
├── render.yaml
└── .env.example
```

---

## Run locally

```bash
# 1. Clone and enter the project
git clone https://github.com/ayush-s-tomar/agentloop.git
cd agentloop

# 2. Create virtual environment (Python 3.11+ required)
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env — add GROQ_API_KEY and TAVILY_API_KEY

# 5. Start the server
uvicorn main:app --reload

# 6. Open http://127.0.0.1:8000
```

Free API keys (no credit card needed):
- Groq → https://console.groq.com/keys
- Tavily → https://app.tavily.com

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/research/start` | Start a background research job → returns `job_id` |
| `GET` | `/api/research/status/{job_id}` | Poll for live trace + completion |
| `POST` | `/api/research` | Synchronous run (returns when complete) |
| `GET` | `/api/history` | List past sessions |
| `GET` | `/api/history/{session_id}` | Fetch one past session in full |

```bash
# Example
curl -X POST https://agentloop.onrender.com/api/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "impact of the EU AI Act on startups"}'
```

---

## Deploy to Render

1. Fork/clone this repo and push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env vars: `GROQ_API_KEY`, `TAVILY_API_KEY`, `GROQ_MODEL=llama-3.3-70b-versatile`, `PYTHON_VERSION=3.11.0`
6. Deploy

> **Note:** Render free tier has ephemeral storage — SQLite memory resets on redeploy. For persistent memory, attach a Render Disk or swap to Postgres.

---

## What makes this agentic

- **Real tool-calling** — the LLM is given a tool schema and decides per sub-question whether and how to call `web_search`. It's not a hardcoded "always search" pipeline. See `agent/llm.py::run_with_tools`.
- **Conditional looping** — LangGraph conditional edges route `research → research` while sub-questions remain, and `reflect → research` if the agent finds gaps in its own notes.
- **Two kinds of memory** — short-term (notes accumulated within one run's state) and long-term (SQLite, persisted across runs, checked via keyword-overlap at the start of every new run).
- **Observability** — every node emits a trace event that streams live to the UI, showing exactly what the agent is doing at each step.

---

## What I'd add next

- Vector-based memory recall (pgvector / Chroma) instead of keyword overlap
- A second tool (calculator, doc retrieval) to show the agent choosing *between* tools
- Token-level streaming within each node for fully real-time output
- Eval harness with LLM-as-judge rubric to catch prompt regressions