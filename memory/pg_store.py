"""Postgres (Supabase) history backend. Same API as the SQLite functions in store.py."""
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import psycopg2
import psycopg2.extras

TABLE = "agentloop_sessions"


def _connect():
    return psycopg2.connect(
        host=os.environ["SUPABASE_DB_HOST"],
        port=os.environ.get("SUPABASE_DB_PORT", "5432"),
        dbname=os.environ.get("SUPABASE_DB_NAME", "postgres"),
        user=os.environ["SUPABASE_DB_USER"],
        password=os.environ["SUPABASE_DB_PASSWORD"],
        sslmode="require",
        connect_timeout=8,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


@contextmanager
def _cursor():
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS " + TABLE + " ("
            "id TEXT PRIMARY KEY, topic TEXT NOT NULL, plan_json TEXT NOT NULL, "
            "notes_json TEXT NOT NULL, report TEXT NOT NULL, "
            "created_at DOUBLE PRECISION NOT NULL, owner TEXT NOT NULL DEFAULT '')"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS " + TABLE + "_owner_idx ON " + TABLE + " (owner, created_at DESC)")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def save_session(session_id, topic, plan, notes, report, owner=""):
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO " + TABLE + " (id, topic, plan_json, notes_json, report, created_at, owner) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET topic=EXCLUDED.topic, plan_json=EXCLUDED.plan_json, "
            "notes_json=EXCLUDED.notes_json, report=EXCLUDED.report, created_at=EXCLUDED.created_at, owner=EXCLUDED.owner",
            (session_id, topic, json.dumps(plan), json.dumps(notes), report, time.time(), owner or ""),
        )


def list_sessions(limit: int = 20, owner: str = "") -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, topic, created_at FROM " + TABLE + " WHERE owner = %s ORDER BY created_at DESC LIMIT %s",
            (owner or "", limit),
        )
        return [dict(r) for r in cur.fetchall()]


def get_session(session_id: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM " + TABLE + " WHERE id = %s", (session_id,))
        row = cur.fetchone()
    if not row or (owner is not None and row["owner"] != owner):
        return None
    d = dict(row)
    d.pop("owner", None)
    d["plan"] = json.loads(d.pop("plan_json"))
    d["notes"] = json.loads(d.pop("notes_json"))
    return d


def delete_session(session_id: str, owner: Optional[str] = None) -> None:
    with _cursor() as cur:
        if owner is None:
            cur.execute("DELETE FROM " + TABLE + " WHERE id = %s", (session_id,))
        else:
            cur.execute("DELETE FROM " + TABLE + " WHERE id = %s AND owner = %s", (session_id, owner))


def delete_all_sessions(owner: Optional[str] = None) -> None:
    with _cursor() as cur:
        if owner is None:
            cur.execute("DELETE FROM " + TABLE)
        else:
            cur.execute("DELETE FROM " + TABLE + " WHERE owner = %s", (owner,))


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def find_similar_session(topic: str, threshold: float = 0.45, owner: str = ""):
    target = _words(topic)
    if not target:
        return None
    with _cursor() as cur:
        cur.execute(
            "SELECT id, topic, report, created_at FROM " + TABLE + " WHERE owner = %s ORDER BY created_at DESC LIMIT 200",
            (owner or "",),
        )
        rows = cur.fetchall()
    best, best_score = None, 0.0
    for row in rows:
        cand = _words(row["topic"])
        if not cand:
            continue
        score = len(target & cand) / len(target | cand)
        if score > best_score:
            best, best_score = dict(row), score
    return best if best and best_score >= threshold else None