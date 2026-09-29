"""SQLite schema and connection management.

Stores metadata and derived fields only (message_id, sender, subject,
timestamp, classification fields, decision fields, call-observability
metrics) — never raw email body content. Gmail (or, in earlier milestones,
a synthetic fixture) remains the source of truth for the original message.

Uses the stdlib `sqlite3` module directly — no ORM. The schema is a single,
deliberately small table: this is a classification cache and decision
audit record, not a general-purpose database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS email_records (
    message_id                 TEXT PRIMARY KEY,
    sender                      TEXT NOT NULL,
    subject                     TEXT NOT NULL,
    email_timestamp             TEXT NOT NULL,

    classification_category     TEXT NOT NULL,
    classification_urgency      TEXT NOT NULL,
    classification_confidence   REAL NOT NULL,
    classification_risk_flags   TEXT NOT NULL,
    classification_summary      TEXT NOT NULL,
    classification_reasoning    TEXT NOT NULL,

    decision_category            TEXT,
    decision_source               TEXT,
    decision_reasoning            TEXT,

    processed_at                  TEXT NOT NULL,
    decided_at                    TEXT,

    -- Observability (Part 1): metrics about the classifier call itself,
    -- never about email content. NULL for any row saved before this
    -- instrumentation existed, or when the classifier implementation
    -- doesn't report them — never fabricated as 0.
    latency_ms                    REAL,
    input_tokens                  INTEGER,
    output_tokens                 INTEGER,
    cost_usd                      REAL,
    retry_count                   INTEGER,
    error_type                    TEXT
);
"""

# Columns added after the original schema above (kept here, rather than only
# inline in SCHEMA, so _ensure_observability_columns can retrofit them onto
# a pre-existing database created before Part 1 — see its docstring).
_OBSERVABILITY_COLUMNS: dict[str, str] = {
    "latency_ms": "REAL",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cost_usd": "REAL",
    "retry_count": "INTEGER",
    "error_type": "TEXT",
}


def _ensure_observability_columns(conn: sqlite3.Connection) -> None:
    """Adds the Part-1 observability columns to an existing `email_records`
    table if they're missing.

    `CREATE TABLE IF NOT EXISTS` (see SCHEMA) is a no-op on a table that
    already exists, even if its columns differ from the current SCHEMA
    text — it does NOT retroactively add new columns. This migration step
    fills that gap: it's additive only (SQLite `ALTER TABLE ... ADD
    COLUMN`, one per missing column), never drops or rewrites existing
    data, and is a no-op on a database that already has every column
    (checked via `PRAGMA table_info`, not by catching the "duplicate
    column" error). Safe to call every time `initialize_database` runs.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}
    for column, sql_type in _OBSERVABILITY_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE email_records ADD COLUMN {column} {sql_type}")


def initialize_database(db_path: str | Path) -> None:
    """Create the database file and schema if they don't already exist,
    and migrate an existing database to have every current column.

    Safe to call multiple times: `CREATE TABLE IF NOT EXISTS` plus the
    additive column migration above mean a second (or third) call leaves
    existing data untouched. Never drops or recreates the database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as conn:
        conn.execute(SCHEMA)
        _ensure_observability_columns(conn)
        conn.commit()


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """A SQLite connection with row access by column name, always closed
    on exit — including when the caller's block raises.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
