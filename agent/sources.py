"""Transparent domain-quality rules for ranking search hits. Judgement, not measurement: edit freely."""
from urllib.parse import urlparse

PRIMARY_SUFFIXES = (".gov", ".edu", ".int", ".mil", ".gov.uk", ".ac.uk", ".europa.eu")
PRIMARY_DOMAINS = {
    "europa.eu", "arxiv.org", "nature.com", "science.org", "who.int", "oecd.org", "imf.org",
    "worldbank.org", "iea.org", "nih.gov", "sec.gov", "bls.gov", "iso.org", "ieee.org", "acm.org",
    "nber.org", "congress.gov", "eur-lex.europa.eu",
}
REPUTABLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "ft.com", "wsj.com", "economist.com",
    "nytimes.com", "bloomberg.com", "wikipedia.org", "mckinsey.com", "brookings.edu", "mit.edu",
}
LOW_DOMAINS = {
    "medium.com", "substack.com", "quora.com", "reddit.com", "linkedin.com", "facebook.com",
    "x.com", "twitter.com", "pinterest.com", "blogspot.com", "wordpress.com", "tumblr.com",
}

BONUS = {"primary": 0.15, "reputable": 0.05, "general": 0.0, "low": -0.2}


def _host(url):
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def _matches(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)


def tier(url):
    """'primary' | 'reputable' | 'general' | 'low' for a URL. Never raises."""
    host = _host(url)
    if not host:
        return "general"
    if _matches(host, LOW_DOMAINS):
        return "low"
    if _matches(host, PRIMARY_DOMAINS) or host.endswith(PRIMARY_SUFFIXES):
        return "primary"
    if _matches(host, REPUTABLE_DOMAINS):
        return "reputable"
    return "general"


def adjusted_score(hit):
    """Tavily score plus the tier bonus; None when the hit has no score."""
    score = hit.get("score")
    if score is None:
        return None
    return score + BONUS[tier(hit.get("url", ""))]