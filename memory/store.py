"""
Long-term memory for the agent.

Short-term memory (the notes gathered during one research run) lives in the
LangGraph state and disappears when the run ends. This module is the agent's
long-term memory: every finished run is written to SQLite, and the recall step
at the start of a new run checks whether the agent already looked into
something similar before.

Sessions carry an `owner` (a random per-browser id sent by the page). Every
read and delete is scoped to that owner, so visitors do not see each other's
history. This is isolation, not authentication.

Dependency-free (stdlib sqlite3) so there is nothing extra to provision.
"""

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agentloop.db"


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                notes_json TEXT NOT NULL,
                report TEXT NOT NULL,
                created_at REAL NOT NULL,
                owner TEXT NOT NULL DEFAULT ''
            )
            """
        )
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "owner" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner TEXT NOT NULL DEFAULT ''")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def save_session(session_id: str, topic: str, plan: List[str], notes: List[dict], report: str, owner: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, topic, plan_json, notes_json, report, created_at, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, topic, json.dumps(plan), json.dumps(notes), report, time.time(), owner or ""),
        )


def list_sessions(limit: int = 20, owner: str = "") -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, topic, created_at FROM sessions WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
            (owner or "", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        if owner is not None and row["owner"] != owner:
            return None
        d = dict(row)
        d.pop("owner", None)
        d["plan"] = json.loads(d.pop("plan_json"))
        d["notes"] = json.loads(d.pop("notes_json"))
        return d


def delete_session(session_id: str, owner: Optional[str] = None) -> None:
    """Delete a single past run by id (only the owner's, when owner is given)."""
    with _connect() as conn:
        if owner is None:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        else:
            conn.execute("DELETE FROM sessions WHERE id = ? AND owner = ?", (session_id, owner))


def delete_all_sessions(owner: Optional[str] = None) -> None:
    """Wipe long-term memory: everything when owner is None, else only that owner's."""
    with _connect() as conn:
        if owner is None:
            conn.execute("DELETE FROM sessions")
        else:
            conn.execute("DELETE FROM sessions WHERE owner = ?", (owner,))


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def find_similar_session(topic: str, threshold: float = 0.45, owner: str = "") -> Optional[Dict[str, Any]]:
    """
    Keyword-overlap (Jaccard) match against this owner's past topics.

    Intentionally simple rather than vector based: dependency-free and easy to
    explain. Swapping in a vector store is a natural next step.
    """
    target = _words(topic)
    if not target:
        return None

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, topic, report, created_at FROM sessions WHERE owner = ? "
            "ORDER BY created_at DESC LIMIT 200",
            (owner or "",),
        ).fetchall()

    best, best_score = None, 0.0
    for row in rows:
        candidate = _words(row["topic"])
        if not candidate:
            continue
        overlap = len(target & candidate) / len(target | candidate)
        if overlap > best_score:
            best, best_score = dict(row), overlap

    if best and best_score >= threshold:
        return best
    return None