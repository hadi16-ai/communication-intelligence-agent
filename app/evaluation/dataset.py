"""Loads the synthetic evaluation dataset (data/eval_dataset.json) as
typed ground-truth records.

This module knows nothing about Gemini, the decision engine, or storage —
it only turns the dataset's raw JSON into validated, typed data built on
the existing EmailMessage/Category models. No network calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.models import Category, EmailMessage

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_dataset.json"


class GroundTruthRecord(BaseModel):
    """One labeled dataset example: an email plus the category a human
    decided it should be classified as.

    This is CLASSIFICATION-level ground truth only — the dataset does not
    independently label urgency or risk flags, and it does not label a
    "correct decision" at all. See app/evaluation/evaluator.py for how an
    "expected decision" is deterministically derived from this, using the
    real DecisionEngine rather than inventing a second ground truth.
    """

    message_id: str = Field(..., min_length=1)
    sender: str = Field(..., min_length=1)
    subject: str = ""
    body: str = ""
    received_at: datetime
    expected_category: Category
    notes: str = ""

    def to_email_message(self) -> EmailMessage:
        return EmailMessage(
            message_id=self.message_id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            received_at=self.received_at,
        )


def load_ground_truth(path: str | Path = DEFAULT_DATASET_PATH) -> list[GroundTruthRecord]:
    """Loads and validates every record in the dataset.

    Raises `pydantic.ValidationError` if a record's shape doesn't match,
    and `json.JSONDecodeError` if the file isn't valid JSON — the same
    "fail clearly rather than silently" posture used throughout the app.
    """
    raw_records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GroundTruthRecord(
            message_id=record["message_id"],
            sender=record["sender"],
            subject=record["subject"],
            body=record["body"],
            received_at=datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")),
            expected_category=Category(record["expected_category"]),
            notes=record.get("notes", ""),
        )
        for record in raw_records
    ]
