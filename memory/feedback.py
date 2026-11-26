"""Thumbs up/down per saved session. SQLite only; resets with the database file."""
import sqlite3
import time
from contextlib import contextmanager

from memory import store


@contextmanager
def _conn():
    store.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(store.DB_PATH)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feedback ("
            "session_id TEXT PRIMARY KEY, owner TEXT NOT NULL, rating INTEGER NOT NULL, created_at REAL NOT NULL)"
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def save(session_id, owner, rating):
    """Store 1 (up) or -1 (down); replaces this session's earlier vote."""
    if rating not in (1, -1):
        raise ValueError("rating must be 1 or -1")
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO feedback (session_id, owner, rating, created_at) VALUES (?, ?, ?, ?)",
            (session_id, owner or "", rating, time.time()),
        )


def get(session_id, owner):
    """This owner's rating for the session: 1, -1, or 0 when none."""
    with _conn() as c:
        row = c.execute(
            "SELECT rating FROM feedback WHERE session_id = ? AND owner = ?", (session_id, owner or "")
        ).fetchone()
    return row[0] if row else 0


def summary():
    with _conn() as c:
        up = c.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
        down = c.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1").fetchone()[0]
    return {"up": up, "down": down}