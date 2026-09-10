"""SQLite schema and connection management.

Stores metadata and derived fields only (message_id, sender, subject,
timestamp, classification fields, decision fields) — never raw email body
content. Gmail (or, in earlier milestones, a synthetic fixture) remains the
source of truth for the original message.

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
    decided_at                    TEXT
);
"""


def initialize_database(db_path: str | Path) -> None:
    """Create the database file and schema if they don't already exist.

    Safe to call multiple times: `CREATE TABLE IF NOT EXISTS` means a
    second (or third) call leaves existing data untouched. Never drops or
    recreates the database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as conn:
        conn.execute(SCHEMA)
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
