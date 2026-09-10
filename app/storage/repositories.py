"""EmailRepository: save/query classifications and decisions, keyed by the
stable Gmail-style `message_id`. This is the caching layer — checking
`has_classification()` before calling the M4 classifier is what avoids
repeat Gemini calls for the same email (wired together in app/core/pipeline
in M7; not done here).

No SQL is ever built by interpolating email- or user-controlled values —
every query uses parameterized placeholders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from app.core.models import (
    Category,
    ClassificationResult,
    Decision,
    DecisionSource,
    EmailMessage,
    RiskFlag,
    UrgencyLevel,
)
from app.storage.database import get_connection, initialize_database


class StoredEmailRecord(BaseModel):
    """A composed read-view over one stored row: email metadata plus its
    classification and (if present) decision. Composes the existing
    models rather than duplicating their fields.
    """

    message_id: str
    sender: str
    subject: str
    received_at: datetime
    classification: ClassificationResult
    decision: Decision | None = None


class StorageError(RuntimeError):
    """Raised when a stored record is malformed/corrupt and cannot be
    reconstructed into the expected Pydantic models. Never silently
    swallowed into a fabricated or partial result — a corrupt row must
    fail loudly rather than produce a wrong classification.
    """


class EmailRepository:
    """SQLite-backed storage and cache for email classifications and
    decisions. One row per `message_id`; re-saving a classification for
    the same message_id updates that row rather than creating a new one.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)

    def initialize(self) -> None:
        """Create the database/schema if needed. Idempotent."""
        initialize_database(self._db_path)

    def save_classification(self, email: EmailMessage, classification: ClassificationResult) -> None:
        """Insert or update the row for this email's classification.

        Deliberately does not touch any existing decision columns — a
        decision already stored for this message_id survives a
        re-classification.
        """
        if classification.message_id != email.message_id:
            raise ValueError(
                f"classification.message_id ({classification.message_id!r}) does not match "
                f"email.message_id ({email.message_id!r})"
            )

        risk_flags_json = json.dumps([flag.value for flag in classification.risk_flags])
        processed_at = datetime.now(timezone.utc).isoformat()

        with get_connection(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO email_records (
                    message_id, sender, subject, email_timestamp,
                    classification_category, classification_urgency,
                    classification_confidence, classification_risk_flags,
                    classification_summary, classification_reasoning,
                    processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    sender = excluded.sender,
                    subject = excluded.subject,
                    email_timestamp = excluded.email_timestamp,
                    classification_category = excluded.classification_category,
                    classification_urgency = excluded.classification_urgency,
                    classification_confidence = excluded.classification_confidence,
                    classification_risk_flags = excluded.classification_risk_flags,
                    classification_summary = excluded.classification_summary,
                    classification_reasoning = excluded.classification_reasoning,
                    processed_at = excluded.processed_at
                """,
                (
                    email.message_id,
                    email.sender,
                    email.subject,
                    email.received_at.isoformat(),
                    classification.category.value,
                    classification.urgency.value,
                    classification.confidence,
                    risk_flags_json,
                    classification.summary,
                    classification.reasoning,
                    processed_at,
                ),
            )
            conn.commit()

    def save_decision(self, decision: Decision) -> None:
        """Attach a decision to an already-stored classification.

        A classification must already exist for `decision.message_id` —
        decisions are always derived from a classification (see
        app/core/decisions.py), so saving one without the other would be
        an inconsistent record.
        """
        with get_connection(self._db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM email_records WHERE message_id = ?",
                (decision.message_id,),
            ).fetchone()
            if exists is None:
                raise ValueError(
                    f"No stored classification found for message_id {decision.message_id!r}; "
                    "save_classification() must be called before save_decision()."
                )

            conn.execute(
                """
                UPDATE email_records
                SET decision_category = ?, decision_source = ?, decision_reasoning = ?, decided_at = ?
                WHERE message_id = ?
                """,
                (
                    decision.category.value,
                    decision.source.value,
                    decision.reasoning,
                    datetime.now(timezone.utc).isoformat(),
                    decision.message_id,
                ),
            )
            conn.commit()

    def has_classification(self, message_id: str) -> bool:
        """Cache check: has this message_id already been classified?"""
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM email_records WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return row is not None

    def get_classification(self, message_id: str) -> ClassificationResult | None:
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM email_records WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_classification(row)

    def get_decision(self, message_id: str) -> Decision | None:
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM email_records WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if row is None or row["decision_category"] is None:
            return None
        return self._row_to_decision(row)

    def get_record(self, message_id: str) -> StoredEmailRecord | None:
        """The full composed view: email metadata + classification +
        decision (if one has been saved).
        """
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM email_records WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_stored_record(row)

    def list_records(self) -> list[StoredEmailRecord]:
        """Every stored record, most recently processed first.

        Used by the dashboard (and anything else that needs to enumerate
        everything rather than look up one message_id). A record whose
        classification exists but hasn't been decided yet is still
        included, with `decision=None` — callers must handle that rather
        than assume every row has a decision.
        """
        with get_connection(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM email_records ORDER BY processed_at DESC").fetchall()
        return [self._row_to_stored_record(row) for row in rows]

    def _row_to_stored_record(self, row) -> StoredEmailRecord:
        try:
            received_at = datetime.fromisoformat(row["email_timestamp"])
        except ValueError as exc:
            raise StorageError(
                f"Stored email_timestamp for message_id {row['message_id']!r} is malformed: {exc}"
            ) from exc

        classification = self._row_to_classification(row)
        decision = self._row_to_decision(row) if row["decision_category"] is not None else None

        return StoredEmailRecord(
            message_id=row["message_id"],
            sender=row["sender"],
            subject=row["subject"],
            received_at=received_at,
            classification=classification,
            decision=decision,
        )

    def _row_to_classification(self, row) -> ClassificationResult:
        try:
            risk_flags_raw = json.loads(row["classification_risk_flags"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise StorageError(
                f"Stored risk_flags for message_id {row['message_id']!r} is not valid JSON: {exc}"
            ) from exc

        try:
            return ClassificationResult(
                message_id=row["message_id"],
                category=Category(row["classification_category"]),
                urgency=UrgencyLevel(row["classification_urgency"]),
                confidence=row["classification_confidence"],
                risk_flags=[RiskFlag(flag) for flag in risk_flags_raw],
                summary=row["classification_summary"],
                reasoning=row["classification_reasoning"],
            )
        except Exception as exc:
            raise StorageError(
                f"Stored classification for message_id {row['message_id']!r} is malformed: {exc}"
            ) from exc

    def _row_to_decision(self, row) -> Decision:
        classification = self._row_to_classification(row)
        try:
            return Decision(
                message_id=row["message_id"],
                category=Category(row["decision_category"]),
                reasoning=row["decision_reasoning"],
                source=DecisionSource(row["decision_source"]),
                classification=classification,
            )
        except Exception as exc:
            raise StorageError(
                f"Stored decision for message_id {row['message_id']!r} is malformed: {exc}"
            ) from exc
