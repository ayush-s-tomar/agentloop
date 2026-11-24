import pytest

from agent.graph import _relevant
from agent.sources import adjusted_score, tier


@pytest.mark.parametrize("url,expected", [
    ("https://www.bls.gov/ooh/", "primary"),
    ("https://eur-lex.europa.eu/eli/reg/2024/1689", "primary"),
    ("https://www.cs.mit.edu/x", "primary"),
    ("https://arxiv.org/abs/1", "primary"),
    ("https://www.reuters.com/a", "reputable"),
    ("https://medium.com/@x/post", "low"),
    ("https://someone.substack.com/p/1", "low"),
    ("https://example.com/page", "general"),
    ("not a url", "general"),
    ("", "general"),
])
def test_tiers(url, expected):
    assert tier(url) == expected


def test_lookalike_domains_do_not_match():
    assert tier("https://notmedium.com/x") == "general"
    assert tier("https://fakegov.com/x") == "general"


def test_adjusted_score_and_missing_score():
    assert adjusted_score({"url": "https://bls.gov/a", "score": 0.5}) == pytest.approx(0.65)
    assert adjusted_score({"url": "https://medium.com/a", "score": 0.5}) == pytest.approx(0.3)
    assert adjusted_score({"url": "https://bls.gov/a"}) is None


def test_primary_outranks_slightly_higher_scored_blog():
    hits = [
        {"url": "https://medium.com/@x/a", "score": 0.80},
        {"url": "https://www.bls.gov/a", "score": 0.70},
        {"url": "https://example.com/a", "score": 0.72},
        {"url": "https://example.org/b", "score": 0.71},
    ]
    urls = [h["url"] for h in _relevant(hits)]
    assert urls[0] == "https://www.bls.gov/a" and "https://medium.com/@x/a" not in urls


def test_floor_keeps_two_sources_even_if_all_low():
    hits = [{"url": "https://medium.com/a", "score": 0.4}, {"url": "https://quora.com/b", "score": 0.3},
            {"url": "https://reddit.com/c", "score": 0.2}]
    assert len(_relevant(hits)) == 2