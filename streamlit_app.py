"""
AgentLoop — Streamlit deployment

Drop-in replacement for the FastAPI + SSE frontend. Reuses agent/graph.py,
agent/llm.py, agent/tools.py, agent/state.py, and memory/store.py UNCHANGED —
this file only replaces main.py + templates/index.html + static/index.html.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this repo to GitHub, create a new app on share.streamlit.io pointing
    at streamlit_app.py, and add GROQ_API_KEY / TAVILY_API_KEY / (optionally)
    GROQ_FAST_MODEL, GROQ_REASON_MODEL, MAX_SUBQUESTIONS in the app's
    "Secrets" panel (TOML format, e.g. GROQ_API_KEY = "sk-...").
"""

import logging
import os

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets → environment
# agent/llm.py and agent/tools.py read os.environ directly, so we mirror
# whatever Streamlit Cloud gives us in st.secrets into the process env
# before importing anything that touches Groq/Tavily at import time.
# ---------------------------------------------------------------------------
for key in (
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "GROQ_FAST_MODEL",
    "GROQ_REASON_MODEL",
    "MAX_SUBQUESTIONS",
):
    if key in st.secrets and key not in os.environ:
        os.environ[key] = str(st.secrets[key])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
log = logging.getLogger(__name__)

from agent.graph import run as agent_run
from memory import store


def _escape_dollars(text: str) -> str:
    """
    Streamlit's markdown renderer treats $...$ as LaTeX math. Reports often
    contain price ranges like "$0.05 to $1.00", which get mangled into
    broken math notation. Escaping every bare $ sidesteps that without
    touching the model's wording.
    """
    return text.replace("$", r"\$")

store.init_db()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AgentLoop", page_icon="🔁", layout="wide")

NODE_LABELS = {
    "recall":     "Checking memory …",
    "planner":    "Planning sub-questions …",
    "research":   "Researching …",
    "reflect":    "Reflecting on findings …",
    "synthesize": "Writing report …",
    "persist":    "Saving to memory …",
}
NODE_ORDER = ["recall", "planner", "research", "reflect", "synthesize", "persist"]

if "report" not in st.session_state:
    st.session_state.report = None
if "report_raw" not in st.session_state:
    st.session_state.report_raw = None
if "topic_ran" not in st.session_state:
    st.session_state.topic_ran = None

# ---------------------------------------------------------------------------
# Sidebar — history (reuses memory/store.py, no new logic)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Past sessions")
    try:
        sessions = store.list_sessions(limit=20)
    except Exception as e:
        sessions = []
        st.error(f"Could not load history: {e}")

    if not sessions:
        st.caption("No research runs yet.")
    else:
        for s in sessions:
            if st.button(s["topic"][:60] or "(untitled)", key=f"hist_{s['id']}", use_container_width=True):
                full = store.get_session(s["id"])
                if full:
                    st.session_state.report_raw = full["report"]
                    st.session_state.report = _escape_dollars(full["report"])
                    st.session_state.topic_ran = full["topic"]

    st.divider()
    st.caption("Backend: LangGraph · Groq · Tavily · SQLite")

    if sessions:
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("Clear history", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Delete all past sessions? This can't be undone.")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    store.delete_all_sessions()
                    st.session_state.confirm_clear = False
                    st.session_state.report = None
                    st.session_state.report_raw = None
                    st.session_state.topic_ran = None
                    st.rerun()
            with col_b:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

# ---------------------------------------------------------------------------
# Main — input
# ---------------------------------------------------------------------------
st.title("🔁 AgentLoop")
st.caption("A multi-step research agent with tool-use and memory.")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. impact of the EU AI Act on startups",
        max_chars=300,
    )
    submitted = st.form_submit_button("Run", type="primary")

if submitted:
    topic = (topic or "").strip()
    if not topic:
        st.error("Topic cannot be empty.")
    else:
        st.session_state.topic_ran = topic
        st.session_state.report = None

        # Pre-create one status line per node so the layout doesn't jump
        # around, and update in place as events arrive — this is the
        # Streamlit equivalent of the old SSE live-trace stepper.
        status_slots = {node: st.empty() for node in NODE_ORDER}
        for node in NODE_ORDER:
            status_slots[node].markdown(f"⚪ {NODE_LABELS[node]}")

        detail_box = st.empty()
        final_report = ""
        error_msg = None

        try:
            for node_name, snapshot in agent_run(topic):
                if node_name in status_slots:
                    status_slots[node_name].markdown(f"🟢 {NODE_LABELS[node_name]}")

                if node_name == "planner":
                    qs = snapshot.get("plan", [])
                    if qs:
                        detail_box.info("Sub-questions:\n" + "\n".join(f"- {q}" for q in qs))
                elif node_name == "research":
                    trace = snapshot.get("trace", [])
                    if trace:
                        detail_box.info(trace[-1]["message"])
                elif node_name == "reflect":
                    decision = snapshot.get("decision", "")
                    if decision == "more_research":
                        detail_box.info("Found gaps — researching follow-up questions.")
                    else:
                        detail_box.info("Notes judged sufficient — writing final report.")
                elif node_name == "synthesize":
                    final_report = snapshot.get("final_report", "")

        except Exception as exc:
            log.error("Agent run failed: %s", exc, exc_info=True)
            error_msg = "An error occurred during research. Please try again in a moment."

        if error_msg:
            st.error(error_msg)
        else:
            detail_box.empty()
            st.session_state.report_raw = final_report
            st.session_state.report = _escape_dollars(final_report)
            st.rerun()

# ---------------------------------------------------------------------------
# Report display
# ---------------------------------------------------------------------------
if st.session_state.report:
    st.divider()
    st.subheader(st.session_state.topic_ran or "Report")
    st.markdown(st.session_state.report)
    st.download_button(
        "Download report (.md)",
        data=st.session_state.report_raw or st.session_state.report,
        file_name="agentloop_report.md",
        mime="text/markdown",
    )