"""
Postgres (Supabase) storage layer for AgentOps.
"""

import os
import time
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def _connect():
    return psycopg2.connect(
        host=os.environ["SUPABASE_DB_HOST"],
        port=os.environ.get("SUPABASE_DB_PORT", "5432"),
        dbname=os.environ.get("SUPABASE_DB_NAME", "postgres"),
        user=os.environ["SUPABASE_DB_USER"],
        password=os.environ["SUPABASE_DB_PASSWORD"],
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def init_db():
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id UUID PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    started_at DOUBLE PRECISION NOT NULL,
                    duration_ms DOUBLE PRECISION NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    estimated_cost_usd DOUBLE PRECISION,
                    status TEXT NOT NULL,
                    error TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id UUID PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    score DOUBLE PRECISION NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    notes TEXT
                )
            """)
        conn.commit()


def new_run_id() -> str:
    return str(uuid.uuid4())[:8]


def insert_trace(
    run_id, agent_name, step_name, started_at, duration_ms,
    input_tokens, output_tokens, estimated_cost_usd, status, error=None,
):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO traces
                   (id, run_id, agent_name, step_name, started_at, duration_ms,
                    input_tokens, output_tokens, estimated_cost_usd, status, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (str(uuid.uuid4()), run_id, agent_name, step_name, started_at,
                 duration_ms, input_tokens, output_tokens, estimated_cost_usd,
                 status, error),
            )
        conn.commit()


def log_eval_score(run_id, agent_name, score, notes=""):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_runs (id, run_id, agent_name, score, created_at, notes)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (str(uuid.uuid4()), run_id, agent_name, score, time.time(), notes),
            )
        conn.commit()


def fetch_traces(limit=500):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM traces ORDER BY started_at DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]


def fetch_eval_runs(agent_name=None, limit=500):
    with _connect() as conn:
        with conn.cursor() as cur:
            if agent_name:
                cur.execute(
                    "SELECT * FROM eval_runs WHERE agent_name = %s ORDER BY created_at DESC LIMIT %s",
                    (agent_name, limit),
                )
            else:
                cur.execute("SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]


@contextmanager
def ensure_db():
    init_db()
    yield
