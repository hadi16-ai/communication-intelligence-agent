"""Tests for app.storage.database: schema creation and connection
management. Uses temporary SQLite files only — the production database
path (Settings.database_path) is never touched by any test.
"""

import sqlite3

import pytest

from app.storage.database import get_connection, initialize_database


def test_fresh_database_initializes_successfully(tmp_path):
    db_path = tmp_path / "fresh.db"

    initialize_database(db_path)

    assert db_path.exists()


def test_tables_exist_after_initialization(tmp_path):
    db_path = tmp_path / "fresh.db"
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        tables = {
            row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert "email_records" in tables


def test_initialization_creates_parent_directories(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "app.db"

    initialize_database(db_path)

    assert db_path.exists()


def test_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.db"

    initialize_database(db_path)
    initialize_database(db_path)
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        tables = {
            row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "email_records" in tables


def test_existing_records_survive_reinitialization(tmp_path):
    db_path = tmp_path / "persist.db"
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO email_records (
                message_id, sender, subject, email_timestamp,
                classification_category, classification_urgency,
                classification_confidence, classification_risk_flags,
                classification_summary, classification_reasoning, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg-1",
                "a@example.com",
                "Subject",
                "2026-01-01T00:00:00+00:00",
                "DIGEST",
                "LOW",
                0.5,
                "[]",
                "summary",
                "reasoning",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    initialize_database(db_path)  # re-initialize; must not wipe data

    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM email_records WHERE message_id = ?", ("msg-1",)).fetchone()

    assert row is not None
    assert row["sender"] == "a@example.com"


def test_connection_is_closed_after_context_manager_exits(tmp_path):
    db_path = tmp_path / "closed.db"
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_connection_closes_even_if_the_caller_raises(tmp_path):
    db_path = tmp_path / "closed_on_error.db"
    initialize_database(db_path)

    captured_conn = None
    with pytest.raises(ValueError):
        with get_connection(db_path) as conn:
            captured_conn = conn
            raise ValueError("boom")

    with pytest.raises(sqlite3.ProgrammingError):
        captured_conn.execute("SELECT 1")


def test_observability_columns_exist_on_a_freshly_created_database(tmp_path):
    db_path = tmp_path / "fresh_observability.db"
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}

    for column in ("latency_ms", "input_tokens", "output_tokens", "cost_usd", "retry_count", "error_type"):
        assert column in columns


def test_migration_adds_observability_columns_to_a_pre_part1_database(tmp_path):
    """A database created before Part 1's instrumentation existed has
    none of the new columns. initialize_database() must retrofit them —
    additively, without dropping the table or losing existing rows —
    the same "safe to call on an existing local DB" guarantee the schema
    has always had.
    """
    db_path = tmp_path / "pre_existing.db"
    old_schema = """
    CREATE TABLE email_records (
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
    conn = sqlite3.connect(str(db_path))
    conn.execute(old_schema)
    conn.execute(
        """
        INSERT INTO email_records (
            message_id, sender, subject, email_timestamp,
            classification_category, classification_urgency,
            classification_confidence, classification_risk_flags,
            classification_summary, classification_reasoning, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pre-existing-msg", "a@example.com", "Subject", "2026-01-01T00:00:00+00:00",
            "DIGEST", "LOW", 0.5, "[]", "summary", "reasoning", "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    initialize_database(db_path)  # migration under test

    with get_connection(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}
        row = conn.execute(
            "SELECT * FROM email_records WHERE message_id = ?", ("pre-existing-msg",)
        ).fetchone()

    for column in ("latency_ms", "input_tokens", "output_tokens", "cost_usd", "retry_count", "error_type"):
        assert column in columns
    # the pre-existing row survives, with NULLs for the new columns rather
    # than being dropped or fabricated
    assert row["sender"] == "a@example.com"
    assert row["latency_ms"] is None
    assert row["retry_count"] is None


def test_migration_is_a_no_op_when_columns_already_exist(tmp_path):
    """Calling initialize_database twice on an already-migrated database
    must not error (e.g. from a naive "ADD COLUMN" without checking first).
    """
    db_path = tmp_path / "already_migrated.db"
    initialize_database(db_path)

    initialize_database(db_path)  # must not raise "duplicate column name"

    with get_connection(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}
    assert "latency_ms" in columns


def test_persists_across_separate_connections(tmp_path):
    """Simulates closing and reopening the database (e.g. across process
    runs): data written in one connection must be visible in another.
    """
    db_path = tmp_path / "reopen.db"
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO email_records (
                message_id, sender, subject, email_timestamp,
                classification_category, classification_urgency,
                classification_confidence, classification_risk_flags,
                classification_summary, classification_reasoning, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg-reopen",
                "a@example.com",
                "Subject",
                "2026-01-01T00:00:00+00:00",
                "DIGEST",
                "LOW",
                0.5,
                "[]",
                "summary",
                "reasoning",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM email_records WHERE message_id = ?", ("msg-reopen",)
        ).fetchone()

    assert row is not None
