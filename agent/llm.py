"""
LLM wrapper supporting both Groq and Google Gemini.

Set LLM_PROVIDER=gemini in your .env to use Gemini (free, generous limits).
Set LLM_PROVIDER=groq (default) to use Groq.

Gemini free tier: 1500 requests/day, 1M tokens/min — effectively unlimited for demos.
Groq free tier: 100k tokens/day — can get exhausted during heavy testing.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

_PROVIDER = None
_client = None


def _get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def _get_client():
    global _client, _PROVIDER
    provider = _get_provider()
    if _client is None or provider != _PROVIDER:
        _PROVIDER = provider
        if provider == "gemini":
            from openai import OpenAI
            _client = OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            from groq import Groq
            _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def _model(for_tools: bool = False) -> str:
    provider = _get_provider()
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    # Groq
    env = os.environ.get("GROQ_MODEL")
    if env:
        return env
    return "llama-3.3-70b-versatile"


def complete(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _get_client().chat.completions.create(
        model=_model(),
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

        # Hit round cap — ask without tools
        messages.append({"role": "user", "content": "Please give your final answer now, in plain text, no more tool calls."})
        resp = client.chat.completions.create(model=_model(for_tools=True), messages=messages, temperature=0.2)
        return resp.choices[0].message.content or "", log

    except Exception as e:
        err_str = str(e)
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
                    f"Search results:\n{context}\n\nAnswer in 3-6 sentences. Be concrete and cite sources."
                )
                try:
                    answer = complete(system, synthesis_prompt)
                    return answer, fallback_log
                except Exception:
                    pass

            try:
                answer = complete(system, user)
                return answer, []
            except Exception as final_e:
                return f"Research step failed: {final_e}", []

        raise
