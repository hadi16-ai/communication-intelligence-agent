"""Validates data/eval_dataset.json: the synthetic email fixtures used by
M4 (classifier tests) and M9 (evaluation harness). This file only checks
that the fixture data itself is well-formed — it does not implement the
evaluation harness (that's app/evaluation, M9).
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.models import Category, EmailMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "eval_dataset.json"

REQUIRED_FIELDS = {"message_id", "sender", "subject", "timestamp", "body", "expected_category"}


def load_raw_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def to_email_message(record: dict) -> EmailMessage:
    """Fixture records use `timestamp`; EmailMessage uses `received_at`."""
    return EmailMessage(
        message_id=record["message_id"],
        sender=record["sender"],
        subject=record["subject"],
        body=record["body"],
        received_at=datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")),
    )


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    return load_raw_dataset()


class TestDatasetStructure:
    def test_dataset_size_within_expected_range(self, dataset):
        assert 30 <= len(dataset) <= 50

    def test_every_record_has_required_fields(self, dataset):
        for record in dataset:
            missing = REQUIRED_FIELDS - record.keys()
            assert not missing, f"{record.get('message_id')} missing fields: {missing}"

    def test_no_duplicate_message_ids(self, dataset):
        ids = [record["message_id"] for record in dataset]
        duplicates = [msg_id for msg_id, count in Counter(ids).items() if count > 1]
        assert duplicates == []

    def test_message_ids_are_non_empty_strings(self, dataset):
        for record in dataset:
            assert isinstance(record["message_id"], str)
            assert record["message_id"].strip() != ""

    def test_expected_category_is_valid_category_value(self, dataset):
        valid_values = {category.value for category in Category}
        for record in dataset:
            assert record["expected_category"] in valid_values, (
                f"{record['message_id']} has invalid expected_category: "
                f"{record['expected_category']!r}"
            )

    def test_timestamps_are_parseable_iso8601(self, dataset):
        for record in dataset:
            # Must not raise.
            datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))


class TestEveryRecordParsesAsEmailMessage:
    def test_all_records_construct_valid_email_message(self, dataset):
        failures = []
        for record in dataset:
            try:
                to_email_message(record)
            except ValidationError as exc:
                failures.append((record["message_id"], str(exc)))
        assert failures == [], f"Records failed EmailMessage validation: {failures}"

    def test_round_trip_preserves_core_fields(self, dataset):
        record = dataset[0]
        email = to_email_message(record)
        assert email.message_id == record["message_id"]
        assert email.sender == record["sender"]
        assert email.subject == record["subject"]
        assert email.body == record["body"]


class TestCategoryCoverage:
    def test_all_four_categories_are_represented(self, dataset):
        counts = Counter(record["expected_category"] for record in dataset)
        for category in Category:
            assert counts[category.value] > 0, f"No fixture for {category.value}"

    def test_category_breakdown_is_reasonably_balanced(self, dataset):
        """No single category should dominate the dataset — a classifier
        that always guesses one category shouldn't score well against it.
        """
        counts = Counter(record["expected_category"] for record in dataset)
        total = len(dataset)
        for category, count in counts.items():
            assert count / total <= 0.5, f"{category} makes up more than half the dataset"


class TestAdversarialCoverage:
    def test_dataset_includes_prompt_injection_style_content(self, dataset):
        injection_markers = ["SYSTEM INSTRUCTION", "ignore all previous instructions", "AI assistant"]
        matches = [
            record
            for record in dataset
            if any(marker.lower() in record["body"].lower() for marker in injection_markers)
        ]
        assert len(matches) >= 2
        # Adversarial content must still be labeled QUARANTINE, not whatever
        # the injected text asks for.
        for record in matches:
            assert record["expected_category"] == "QUARANTINE"

    def test_dataset_includes_ambiguous_cases_via_notes(self, dataset):
        ambiguous = [r for r in dataset if "ambiguous" in r.get("notes", "").lower()]
        assert len(ambiguous) >= 3
