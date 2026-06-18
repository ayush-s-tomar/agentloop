# AgentLoop

A small, real agentic AI project: a research agent that **plans**, **calls tools**,
**reflects on its own gaps**, and **remembers past runs** — built to demonstrate
multi-step reasoning and tool-use, not just a single LLM call wrapped in a UI.

Give it a topic. It will:

1. **Recall** — check long-term memory (SQLite) for similar research it already did
2. **Plan** — break the topic into a handful of specific sub-questions
3. **Research** — for each sub-question, the LLM *decides* whether to call a live
   web-search tool (Tavily), reads the results, and writes a short cited answer
4. **Reflect** — re-reads its own notes and decides if there are gaps; if so, loops
   back and researches more (bounded, so it can't loop forever)
5. **Synthesize** — writes a structured markdown report from everything it gathered
6. **Persist** — saves the run to long-term memory for next time

```
START -> recall -> planner -> research <-+
                                  |       | (loop while sub-questions remain)
                                  v       |
                               reflect ---+ (loop back to research if gaps found)
                                  |
                                  v
                            synthesize -> persist -> END
```

## Why this project exists

A portfolio of single-shot projects (input → model → output) doesn't show an
interviewer that you can build something that makes *decisions across multiple
steps*. This project is deliberately built to demonstrate, concretely:

- **Tool use that's real, not staged** — the LLM is given a tool schema and
  decides on its own whether/how to call `web_search` (see `agent/llm.py::run_with_tools`
  and `agent/tools.py`). It isn't a hardcoded "always call search" pipeline.
- **Multi-step control flow** — a LangGraph `StateGraph` with conditional edges
  that loop (`research → research` while sub-questions remain; `reflect → research`
  if the agent judges its own notes incomplete).
- **Two distinct kinds of memory** — short-term (the `notes` list inside one run's
  state) and long-term (SQLite, persisted across runs and process restarts, with
  a `recall` step that checks for related past research before starting).
- **Observability** — every node emits a trace event, streamed live to the UI over
  Server-Sent Events, so you can literally watch the agent's reasoning unfold.
- **Graceful degradation** — JSON-parsing of LLM output always has a fallback path
  so one malformed response can't crash a run.

## Repository structure

```
agentloop/
├── main.py                 FastAPI app (REST + SSE streaming endpoint)
├── agent/
│   ├── state.py             the AgentState schema shared across all graph nodes
│   ├── graph.py              the LangGraph StateGraph: nodes + routing
│   ├── llm.py                 Groq wrapper: plain completions + the tool-calling loop
│   └── tools.py                web_search tool (Tavily) + its function schema
├── memory/
│   └── store.py              SQLite long-term memory (save/recall past sessions)
├── static/
│   └── index.html            single-page demo UI (pipeline view + live trace)
├── requirements.txt
├── Dockerfile
├── render.yaml               Render Blueprint (optional one-click deploy)
└── .env.example
```

## Prerequisites

- Python 3.11+
- A free **Groq** API key — https://console.groq.com/keys (the LLM)
- A free **Tavily** API key — https://app.tavily.com (the web-search tool)

Both have generous free tiers; no credit card needed to get started.

## Run it locally

```bash
# 1. unzip the project and cd into it
cd agentloop

# 2. create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. configure your API keys
cp .env.example .env
# now open .env and paste in your GROQ_API_KEY and TAVILY_API_KEY

# 5. run the app
uvicorn main:app --reload

# 6. open the demo UI
# visit http://127.0.0.1:8000 in your browser
```

Type a topic (e.g. *"impact of the EU AI Act on startups"*) and hit **Run**. You'll
see the pipeline stepper light up node by node, a live trace of what the agent is
doing (including which search queries it's actually issuing), and the final report
render on the right once it's done.

## API reference

| Method | Path                       | Description                                      |
|--------|----------------------------|---------------------------------------------------|
| POST   | `/api/research`            | Run the agent synchronously, get the final result |
| GET    | `/api/research/stream?topic=...` | Same run, streamed as Server-Sent Events       |
| GET    | `/api/history`             | List past sessions (long-term memory)            |
| GET    | `/api/history/{session_id}`| Fetch one past session in full                   |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "current state of solid-state batteries"}'
```

## Deploying to Render

These steps take you from local code to a public URL.

**1. Push the code to GitHub**

```bash
cd agentloop
git init
git add .
git commit -m "Initial commit: AgentLoop research agent"
gh repo create agentloop --public --source=. --push
# (no GitHub CLI? create a repo on github.com, then:)
# git remote add origin https://github.com/<you>/agentloop.git
# git push -u origin main
```

**2. Create the service on Render**

- Go to https://dashboard.render.com → **New** → **Web Service**
- Connect your GitHub account and select the `agentloop` repo
- Render should auto-detect Python; if asked, set:
  - **Build command:** `pip install -r requirements.txt`
  - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Choose the **Free** instance type (plenty for a demo)

*Alternative:* if you'd rather not click through settings, use the included
`render.yaml` — in the Render dashboard choose **New → Blueprint**, point it at
your repo, and it will read the config automatically.

**3. Add your environment variables**

In the service's **Environment** tab, add:

| Key                      | Value                                  |
|---------------------------|----------------------------------------|
| `GROQ_API_KEY`             | your Groq key                          |
| `TAVILY_API_KEY`           | your Tavily key                        |
| `GROQ_MODEL`               | `llama-3.3-70b-versatile` (or your choice) |
| `MAX_SUBQUESTIONS`         | `4`                                     |
| `MAX_REFLECT_ITERATIONS`   | `2`                                     |

**4. Deploy**

Click **Create Web Service** (or **Apply** for a Blueprint). Render will build
and deploy automatically; once it's live you'll get a URL like
`https://agentloop.onrender.com`. Every future `git push` redeploys it.

**A note on memory persistence:** Render's free tier disk is ephemeral, so the
SQLite long-term memory resets on redeploys/restarts — that's expected for a
demo. For real persistence, attach a paid Render Disk (there's a commented-out
`disk:` block in `render.yaml`) or swap `memory/store.py` for a hosted Postgres
database.

*Docker alternative:* the included `Dockerfile` works as-is on Render (choose
**Docker** as the environment instead of Python), or on Railway, Fly.io, or any
container host — `docker build -t agentloop . && docker run -p 8000:8000 --env-file .env agentloop`.

## Things worth saying about this project in an interview

- *"The agent doesn't always call the search tool the same way — the model
  decides per sub-question whether and how to query, via real OpenAI-style
  function calling against Groq."*
- *"It distinguishes short-term memory, the notes gathered within one run, from
  long-term memory, a SQLite store the agent checks at the start of every new
  run to avoid re-researching something it already covered."*
- *"The reflect step is what makes it agentic rather than a fixed pipeline: the
  model critiques its own output and the graph conditionally routes back into
  research if it finds gaps, capped so it can't loop forever."*
- *"Every node emits a trace event streamed over SSE, which is what backs the
  live pipeline view in the demo — that observability is what you'd build on to
  add logging, evals, or human-in-the-loop checkpoints in a production version."*

## Natural next steps (good follow-up answers if asked "what would you add?")

- Swap the keyword-overlap memory recall (`memory/store.py::find_similar_session`)
  for embeddings + a vector store (e.g. pgvector, Chroma) for fuzzier recall
- Add a second tool (e.g. a calculator, or a doc-retrieval tool over your own
  files) to show the agent choosing *between* multiple tools, not just one
- Stream the LLM's tokens within each node, not just the trace events, for a
  fully real-time feel
- Add a lightweight eval harness: a fixed set of topics + an LLM-as-judge
  rubric to catch regressions when you change prompts
