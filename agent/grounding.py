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

_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "they", "their", "which", "would",
    "could", "should", "about", "than", "then", "also", "into", "such", "these", "those", "there",
    "while", "where", "when", "what", "will", "more", "most", "some", "other", "both", "each",
}
_WORD = re.compile(r"[a-z0-9]+")
_GAP = re.compile(r"\b(notes?|sources?)\b.*\b(not|no|lack|lacks|without)\b|\b(not|no)\b.*\b(notes?|sources?)\b", re.I)


def _keys(text):
    """Content-word keys (first 5 letters, so simple inflections match)."""
    return {w[:5] for w in _WORD.findall((text or "").lower()) if len(w) >= 4 and w not in _STOP and not w.isdigit()}


def check_claims(report, notes, topic="", recalled=None, min_words=6, threshold=0.6):
    """Flag report sentences whose key words mostly do not appear in the notes.

    Lexical heuristic: it catches sentences that drift away from the notes, not
    wrong claims that reuse note vocabulary. Never raises.
    """
    try:
        parts = [topic or "", recalled or ""]
        for n in notes or []:
            parts.append("%s %s" % (n.get("question", ""), n.get("answer", "")))
        known = _keys(" ".join(parts))
        checked, weak = 0, []
        for line in (report or "").split("\n"):
            line = re.sub(r"^[\s#>*\-|\d.]+", "", line).replace("**", "").strip()
            for sent in re.split(r"(?<=[.!?])\s+", line):
                words = _keys(sent)
                if len(words) < min_words or _GAP.search(sent):
                    continue
                checked += 1
                if len(words & known) / len(words) < threshold:
                    weak.append(sent[:160])
        return {"sentences": checked, "weak": weak[:5], "weak_count": len(weak)}
    except Exception:
        return {}