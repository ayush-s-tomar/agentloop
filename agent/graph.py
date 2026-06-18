"""
The agent's control flow, built as a LangGraph StateGraph.

    START
      |
    recall          -> check long-term memory for a similar past run
      |
    planner         -> LLM breaks the topic into sub-questions
      |
    research  <----+  -> for each sub-question: LLM decides to call the
      |            |     web_search tool, reads the results, writes a
      |            |     short cited answer (this is the actual tool-use)
      | (loop while plan has items)
      v
    reflect   ------+  -> LLM checks notes for gaps; either adds more
      |                  sub-questions (loops back to research) or moves on
      v
    synthesize      -> LLM writes the final structured report
      |
    persist         -> save the run to SQLite (long-term memory)
      |
     END

Every node returns only the slice of state it changed; LangGraph merges
that into the running state dict, which is how "memory" of earlier steps
(notes, recalled context, the trace) survives all the way to the final
report without any extra wiring.
"""

import json
import os
from typing import Dict, List

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent import llm, tools
from memory import store


def _trace(state: AgentState, node: str, message: str) -> List[dict]:
    t = list(state.get("trace", []))
    t.append({"node": node, "message": message})
    return t


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def recall_node(state: AgentState) -> dict:
    topic = state["topic"]
    past = store.find_similar_session(topic)
    if past:
        context = f"On {past['created_at']:.0f}, a similar topic (\"{past['topic']}\") was researched. Prior summary:\n{past['report'][:1200]}"
        msg = f"Found related prior research: \"{past['topic']}\" — using it as background context."
    else:
        context = None
        msg = "No related prior research found in long-term memory. Starting fresh."

    return {
        "recalled_context": context,
        "trace": _trace(state, "recall", msg),
    }


PLANNER_SYSTEM = """You are the planning module of a research agent.
Given a topic, break it into a small number of specific, non-overlapping
sub-questions that together would produce a thorough research report.
Respond ONLY with JSON: {"questions": ["...", "..."]}.
"""


def planner_node(state: AgentState) -> dict:
    max_q = int(os.environ.get("MAX_SUBQUESTIONS", "4"))
    user_prompt = f"Topic: {state['topic']}\nProduce at most {max_q} sub-questions."
    if state.get("recalled_context"):
        user_prompt += f"\n\nBackground from prior research (avoid repeating what this already covers):\n{state['recalled_context']}"

    questions: List[str] = []
    try:
        raw = llm.complete(PLANNER_SYSTEM, user_prompt, json_mode=True)
        parsed = json.loads(raw)
        questions = [q.strip() for q in parsed.get("questions", []) if q.strip()][:max_q]
    except Exception:
        pass

    if not questions:
        # Deterministic fallback so a single bad LLM response can't kill the run.
        questions = [f"What is {state['topic']} and why does it matter?",
                     f"What are the most important recent developments in {state['topic']}?"]

    msg = "Planned " + str(len(questions)) + " sub-questions: " + "; ".join(questions)
    return {
        "plan": questions,
        "notes": [],
        "completed_questions": [],
        "reflect_iterations": 0,
        "trace": _trace(state, "planner", msg),
    }


RESEARCH_SYSTEM = """You are the research module of an agent. You will be given
one specific question. Use the web_search tool (you may call it more than once
with different queries if needed) to find current, reliable information, then
answer the question in 3-6 sentences. Be concrete and specific. Do not invent
facts you did not find via search."""


def research_node(state: AgentState) -> dict:
    plan = list(state.get("plan", []))
    if not plan:
        return {}

    question = plan.pop(0)
    completed = list(state.get("completed_questions", [])) + [question]

    captured_sources: List[str] = []

    def executor(args: dict) -> List[dict]:
        query = args.get("query", question)
        results = tools.web_search(query)
        captured_sources.extend(r["url"] for r in results if r.get("url"))
        return results

    answer, tool_log = llm.run_with_tools(
        system=RESEARCH_SYSTEM,
        user=question,
        tools=[tools.WEB_SEARCH_TOOL_SCHEMA],
        tool_executors={"web_search": executor},
    )

    sources = list(dict.fromkeys(captured_sources))[:5]
    note = {"question": question, "answer": answer.strip(), "sources": sources}
    notes = list(state.get("notes", [])) + [note]

    trace = _trace(state, "research", f"Researching: {question}")
    for call in tool_log:
        trace = trace + [{"node": "research", "message": f"  -> called web_search(query=\"{call['args'].get('query', '')}\")"}]

    return {
        "plan": plan,
        "notes": notes,
        "completed_questions": completed,
        "trace": trace,
    }


def route_after_research(state: AgentState) -> str:
    return "research" if state.get("plan") else "reflect"


REFLECT_SYSTEM = """You are the reflection module of a research agent.
You will be given a topic and the notes gathered so far. Decide whether the
notes are sufficient for a thorough report, or whether there are important
gaps. Respond ONLY with JSON:
{"sufficient": true|false, "additional_questions": ["..."]}
additional_questions should contain at most 2 items, only used if sufficient
is false."""


def reflect_node(state: AgentState) -> dict:
    iterations = state.get("reflect_iterations", 0)
    max_iterations = state.get("max_reflect_iterations", 2)

    if iterations >= max_iterations:
        return {
            "decision": "done",
            "trace": _trace(state, "reflect", "Reached reflection cap — moving to synthesis."),
        }

    notes_summary = "\n".join(f"- Q: {n['question']}\n  A: {n['answer']}" for n in state.get("notes", []))
    user_prompt = f"Topic: {state['topic']}\n\nNotes so far:\n{notes_summary}"

    sufficient, additional = True, []
    try:
        raw = llm.complete(REFLECT_SYSTEM, user_prompt, json_mode=True)
        parsed = json.loads(raw)
        sufficient = bool(parsed.get("sufficient", True))
        additional = [q.strip() for q in parsed.get("additional_questions", []) if q.strip()][:2]
    except Exception:
        pass

    if sufficient or not additional:
        return {
            "decision": "done",
            "trace": _trace(state, "reflect", "Notes judged sufficient — moving to synthesis."),
        }

    msg = "Found gaps — adding follow-up questions: " + "; ".join(additional)
    return {
        "decision": "more_research",
        "plan": additional,
        "reflect_iterations": iterations + 1,
        "trace": _trace(state, "reflect", msg),
    }


def route_after_reflect(state: AgentState) -> str:
    return "research" if state.get("decision") == "more_research" else "synthesize"


SYNTHESIZE_SYSTEM = """You are the writing module of a research agent.
Combine the supplied notes into a well-structured markdown report on the
topic: clear headings, a short intro, one section per theme (not necessarily
one per sub-question), and a closing "Sources" section listing every unique
URL referenced. Write in clear prose. Do not fabricate information beyond
what the notes contain."""


def synthesize_node(state: AgentState) -> dict:
    notes = state.get("notes", [])
    notes_block = "\n\n".join(
        f"Q: {n['question']}\nA: {n['answer']}\nSources: {', '.join(n['sources']) or 'none'}"
        for n in notes
    )
    user_prompt = f"Topic: {state['topic']}\n\nNotes:\n{notes_block}"
    if state.get("recalled_context"):
        user_prompt += f"\n\nRelevant background from earlier research:\n{state['recalled_context']}"

    report = llm.complete(SYNTHESIZE_SYSTEM, user_prompt)

    return {
        "final_report": report,
        "trace": _trace(state, "synthesize", "Final report written."),
    }


def persist_node(state: AgentState) -> dict:
    store.save_session(
        session_id=state["session_id"],
        topic=state["topic"],
        plan=state.get("completed_questions", []),
        notes=state.get("notes", []),
        report=state.get("final_report", ""),
    )
    return {"trace": _trace(state, "persist", "Saved this run to long-term memory.")}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("recall", recall_node)
    g.add_node("planner", planner_node)
    g.add_node("research", research_node)
    g.add_node("reflect", reflect_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("persist", persist_node)

    g.add_edge(START, "recall")
    g.add_edge("recall", "planner")
    g.add_edge("planner", "research")
    g.add_conditional_edges("research", route_after_research, {"research": "research", "reflect": "reflect"})
    g.add_conditional_edges("reflect", route_after_reflect, {"research": "research", "synthesize": "synthesize"})
    g.add_edge("synthesize", "persist")
    g.add_edge("persist", END)

    return g.compile()


graph = build_graph()
