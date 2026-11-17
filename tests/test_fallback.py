from types import SimpleNamespace

import pytest

from agent import llm


class FakeRate(Exception):
    pass


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setattr(llm, "RateLimitError", FakeRate)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "REASON_MODEL", "big")
    monkeypatch.setattr(llm, "FAST_MODEL", "small")
    monkeypatch.delenv("GROQ_FALLBACK_MODEL", raising=False)
    llm._stats["fallbacks"] = 0


def client(limited):
    seen = []

    def create(model, **kw):
        seen.append(model)
        if model in limited:
            raise FakeRate()
        msg = SimpleNamespace(content=" ok ")
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=msg)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return fake, seen


def test_reasoning_model_falls_back_to_fast_model(monkeypatch):
    fake, seen = client({"big"})
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    assert llm._call([{"role": "user", "content": "x"}], "big", 10) == "ok"
    assert seen == ["big", "big", "big", "small"]
    assert llm.fallback_count() == 1


def test_fast_model_has_no_default_fallback(monkeypatch):
    fake, seen = client({"small"})
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    with pytest.raises(FakeRate):
        llm._call([{"role": "user", "content": "x"}], "small", 10)
    assert len(seen) == llm.MAX_RETRIES and llm.fallback_count() == 0


def test_env_fallback_model_is_used(monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "other")
    fake, seen = client({"small"})
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    assert llm._call([{"role": "user", "content": "x"}], "small", 10) == "ok"
    assert seen[-1] == "other"


def test_fails_when_fallback_is_also_limited(monkeypatch):
    fake, seen = client({"big", "small"})
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    with pytest.raises(FakeRate):
        llm._call([{"role": "user", "content": "x"}], "big", 10)
    assert len(seen) == llm.MAX_RETRIES