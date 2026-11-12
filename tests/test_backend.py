import memory.store as store


def test_default_backend_is_sqlite(monkeypatch):
    monkeypatch.delenv("HISTORY_BACKEND", raising=False)
    assert store._select_backend() == "sqlite"


def test_postgres_needs_credentials(monkeypatch):
    monkeypatch.setenv("HISTORY_BACKEND", "postgres")
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    assert store._select_backend() == "sqlite"


def test_postgres_selected_when_configured(monkeypatch):
    monkeypatch.setenv("HISTORY_BACKEND", "postgres")
    for v in ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD"):
        monkeypatch.setenv(v, "x")
    assert store._select_backend() == "postgres"