import pytest

from agent import tools


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("SEARCH_CACHE_DIR", raising=False)
    tools._CACHE.clear()
    yield
    tools._CACHE.clear()


def counting(monkeypatch, results):
    calls = []
    monkeypatch.setattr(tools, "_search_uncached", lambda q, n=5: calls.append(q) or results)
    return calls


def test_disk_cache_replays_after_memory_cache_is_cleared(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))
    calls = counting(monkeypatch, [{"url": "u", "content": "c"}])
    first = tools.web_search("EU AI Act")
    tools._CACHE.clear()
    second = tools.web_search("  eu ai act ")
    assert len(calls) == 1 and first == second


def test_empty_results_are_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))
    calls = counting(monkeypatch, [])
    tools.web_search("q")
    tools._CACHE.clear()
    tools.web_search("q")
    assert len(calls) == 2 and list(tmp_path.iterdir()) == []


def test_corrupt_cache_file_falls_back_to_live_search(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))
    path = tools._disk_path("q", 5)
    path.write_text("not json", encoding="utf-8")
    calls = counting(monkeypatch, [{"url": "u"}])
    assert tools.web_search("q") == [{"url": "u"}] and len(calls) == 1


def test_disk_cache_is_off_by_default(monkeypatch, tmp_path):
    assert tools._disk_path("q", 5) is None
    calls = counting(monkeypatch, [{"url": "u"}])
    tools.web_search("q")
    assert len(calls) == 1