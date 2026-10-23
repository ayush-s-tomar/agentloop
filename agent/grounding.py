"""Check that numbers in a report also appear in the research notes.

This is a cheap guardrail against invented figures. It compares the report to
the notes the agent wrote, not to the original web pages, so a figure that is
wrong in the notes still passes. List numbering and small integers (1-10) are
ignored because they are rarely claims.
"""
import re

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:#+\s*)?\d+\.\s+")


def extract_numbers(text):
    """Return numeric tokens from text, with thousands separators removed."""
    text = _LIST_MARKER.sub("", text or "")
    return [tok.replace(",", "") for tok in _NUM.findall(text)]


def _trivial(tok):
    return tok.isdigit() and int(tok) <= 10


def check_report(report, notes, topic="", recalled=None):
    """Return {"checked": n, "missing": [...], "missing_count": m}. Never raises."""
    try:
        parts = [topic or "", recalled or ""]
        for n in notes or []:
            parts.append("%s %s" % (n.get("question", ""), n.get("answer", "")))
        known = set(extract_numbers(" ".join(parts)))
        seen, missing = set(), []
        for tok in extract_numbers(report or ""):
            if tok in seen or _trivial(tok):
                continue
            seen.add(tok)
            if tok not in known:
                missing.append(tok)
        return {"checked": len(seen), "missing": missing[:10], "missing_count": len(missing)}
    except Exception:
        return {}