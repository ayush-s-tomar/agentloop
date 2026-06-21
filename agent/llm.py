"""
LLM wrapper around Groq with:
  - Model routing  : cheap fast model for tool-heavy research steps,
                     larger model for planning / reflection / synthesis
  - Retry / backoff: up to MAX_RETRIES attempts with exponential back-off
                     that respects the Retry-After header from Groq
  - Token budgets  : kept small to stay within free-tier TPM limits
"""
from __future__ import annotations
import os
import time
import logging
from groq import Groq, RateLimitError, APIStatusError

log = logging.getLogger(__name__)

# ---------- model aliases ----------
FAST_MODEL   = "llama-3.1-8b-instant"   # high TPM limit on free tier — use for research loops
REASON_MODEL = "llama3-70b-8192"         # smarter  — use for plan / reflect / synthesise

MAX_RETRIES    = 6
BASE_DELAY_SEC = 1.5   # minimum wait between retries
MAX_DELAY_SEC  = 60

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
            max_retries=0,   # we handle retries ourselves to get proper back-off
        )
    return _client


def _call(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    client = _get_client()
    delay = BASE_DELAY_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()

        except RateLimitError as exc:
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after", 0))
            except Exception:
                pass
            wait = max(retry_after or delay, delay)
            log.warning(
                "Rate-limited (attempt %d/%d). Waiting %.1fs …",
                attempt, MAX_RETRIES, wait,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY_SEC)

        except APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                log.warning("Groq %d error, retrying in %.1fs …", exc.status_code, delay)
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY_SEC)
            else:
                raise

    raise RuntimeError("Unreachable")


# ---------- public helpers ----------

def plan(topic: str, prior_notes: list[str]) -> list[str]:
    """Return 3–4 sub-questions to research."""
    prior = "\n".join(prior_notes) if prior_notes else "None"
    content = _call(
        messages=[
            {"role": "system", "content": (
                "You are a research planner. Given a topic and any prior notes, "
                "return ONLY a JSON array of 3-4 focused sub-questions (strings). "
                "No markdown, no preamble — raw JSON array only."
            )},
            {"role": "user", "content": f"Topic: {topic}\n\nPrior notes:\n{prior}"},
        ],
        model=REASON_MODEL,
        max_tokens=300,
    )
    import json, re
    # strip accidental markdown fences
    clean = re.sub(r"```[a-z]*", "", content).strip().strip("`").strip()
    return json.loads(clean)


def should_search(question: str) -> bool:
    """Decide whether a question requires a live web search."""
    answer = _call(
        messages=[
            {"role": "system", "content": (
                "Answer with exactly one word: YES if this question requires "
                "current/factual web data, NO if it can be answered from general knowledge."
            )},
            {"role": "user", "content": question},
        ],
        model=FAST_MODEL,
        max_tokens=5,
    )
    return answer.strip().upper().startswith("Y")


def answer_from_context(question: str, search_results: str) -> str:
    """Synthesise a concise answer from raw search snippets."""
    return _call(
        messages=[
            {"role": "system", "content": (
                "You are a research assistant. Using ONLY the provided search results, "
                "write a concise 2-4 sentence answer. Cite no sources by URL — just facts."
            )},
            {"role": "user", "content": f"Question: {question}\n\nSearch results:\n{search_results}"},
        ],
        model=FAST_MODEL,
        max_tokens=350,
    )


def answer_from_knowledge(question: str) -> str:
    """Answer a question from parametric knowledge (no search)."""
    return _call(
        messages=[
            {"role": "system", "content": "Answer the question concisely in 2-4 sentences."},
            {"role": "user", "content": question},
        ],
        model=FAST_MODEL,
        max_tokens=300,
    )


def reflect(topic: str, notes: list[str]) -> tuple[bool, str]:
    """
    Reflect on the notes collected so far.
    Returns (needs_more_research: bool, reflection_text: str).
    """
    notes_text = "\n".join(f"- {n}" for n in notes)
    raw = _call(
        messages=[
            {"role": "system", "content": (
                "You are a critical research editor. "
                "Given a topic and research notes, decide if the notes are sufficient for a report. "
                "Reply with a JSON object: {\"sufficient\": true/false, \"gaps\": \"...\"}. "
                "No markdown, raw JSON only."
            )},
            {"role": "user", "content": f"Topic: {topic}\n\nNotes:\n{notes_text}"},
        ],
        model=REASON_MODEL,
        max_tokens=200,
    )
    import json, re
    clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
    data = json.loads(clean)
    sufficient = bool(data.get("sufficient", True))
    gaps = data.get("gaps", "")
    return (not sufficient), gaps


def synthesise(topic: str, notes: list[str]) -> str:
    """Write a structured Markdown research report from notes."""
    notes_text = "\n".join(f"- {n}" for n in notes)
    return _call(
        messages=[
            {"role": "system", "content": (
                "You are a senior research analyst. "
                "Write a well-structured research report in Markdown with: "
                "## Overview, ## Key Findings (bullet points), ## Analysis, ## Conclusion. "
                "Base it ONLY on the provided notes. Be clear and professional."
            )},
            {"role": "user", "content": f"Topic: {topic}\n\nResearch notes:\n{notes_text}"},
        ],
        model=REASON_MODEL,
        max_tokens=900,
        temperature=0.4,
    )
