"""
Thin wrapper around the Groq chat completions API.

Two entry points:
  - complete()       plain prompt -> text. Used for planning, reflecting,
                      and writing the final report — steps where the model
                      doesn't need any tool.
  - run_with_tools()  the actual function-calling loop. The model is given
                      tool schemas and can choose to call them; we execute
                      the call, feed the result back, and let the model
                      keep going until it answers in plain text (or we hit
                      a round cap, since an agent that can call its own
                      tools needs a hard stop to avoid looping forever).

Kept as raw Groq SDK calls (rather than a framework wrapper) so the
tool-calling mechanics are fully visible — this is the part of the code
worth walking through in an interview.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from groq import Groq

_client: Optional[Groq] = None

# Best Groq model for tool-calling. Override via GROQ_MODEL in .env if needed.
_TOOL_MODEL = "llama-3.3-70b-versatile"
# Lightweight model for plain text steps (planning, reflection, synthesis).
_TEXT_MODEL = "llama-3.3-70b-versatile"


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _model(for_tools: bool = False) -> str:
    env_override = os.environ.get("GROQ_MODEL")
    if env_override:
        return env_override
    return _TOOL_MODEL if for_tools else _TEXT_MODEL


def complete(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _get_client().chat.completions.create(
        model=_model(for_tools=False),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def run_with_tools(
    system: str,
    user: str,
    tools: List[dict],
    tool_executors: Dict[str, Callable[[dict], object]],
    max_rounds: int = 3,
) -> Tuple[str, List[dict]]:
    """
    Returns (final_text, tool_call_log).
    tool_call_log is a list of {"name": ..., "args": ..., "result_preview": ...}
    purely for the trace shown to the user — it's how the demo proves the
    model actually decided to call a tool rather than us faking it.

    Includes a graceful fallback: if the model returns a 400 error on tool-
    calling (e.g. model doesn't support it), we fall back to a direct web
    search using the user question as the query, then synthesize the results.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    log: List[dict] = []

    try:
        for _ in range(max_rounds):
            resp = client.chat.completions.create(
                model=_model(for_tools=True),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or "", log

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                executor = tool_executors.get(name)
                if executor is None:
                    result = {"error": f"no such tool: {name}"}
                else:
                    try:
                        result = executor(args)
                    except Exception as exc:
                        result = {"error": str(exc)}

                log.append({"name": name, "args": args, "result_preview": str(result)[:300]})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)[:4000],
                    }
                )

        # Hit the round cap — ask once more without tools
        messages.append({"role": "user", "content": "Please give your final answer now, in plain text, no more tool calls."})
        resp = client.chat.completions.create(model=_model(for_tools=True), messages=messages, temperature=0.2)
        return resp.choices[0].message.content or "", log

    except Exception as e:
        err_str = str(e)
        # Graceful fallback: if tool-calling fails (e.g. 400 invalid_request_error),
        # run the search directly ourselves then ask the model to synthesise the results.
        if "tool" in err_str.lower() or "400" in err_str or "function" in err_str.lower():
            fallback_log: List[dict] = []
            search_executor = tool_executors.get("web_search")
            search_results = []
            if search_executor:
                try:
                    search_results = search_executor({"query": user})
                    fallback_log.append({"name": "web_search", "args": {"query": user}, "result_preview": str(search_results)[:300]})
                except Exception:
                    pass

            if search_results:
                context = "\n\n".join(
                    f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')}"
                    for r in search_results[:3]
                )
                synthesis_prompt = (
                    f"Using these search results, answer the question: {user}\n\n"
                    f"Search results:\n{context}\n\n"
                    "Answer in 3-6 sentences. Be concrete and cite the sources."
                )
                try:
                    answer = complete(system, synthesis_prompt)
                    return answer, fallback_log
                except Exception:
                    pass

            # Last resort: answer from model's own knowledge
            try:
                answer = complete(system, user)
                return answer, []
            except Exception as final_e:
                return f"Research step failed: {final_e}", []

        raise
