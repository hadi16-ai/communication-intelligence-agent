"""Tests for app.evaluation.dataset: loading data/eval_dataset.json as
typed GroundTruthRecord objects. No network, no Gemini.
"""

import json

import pytest
from pydantic import ValidationError

from app.core.models import Category, EmailMessage
from app.evaluation.dataset import DEFAULT_DATASET_PATH, GroundTruthRecord, load_ground_truth


class TestLoadGroundTruth:
    def test_loads_the_shipped_dataset(self):
        records = load_ground_truth()

        assert len(records) == 41
        assert all(isinstance(r, GroundTruthRecord) for r in records)

    def test_default_path_points_at_the_real_dataset_file(self):
        assert DEFAULT_DATASET_PATH.name == "eval_dataset.json"
        assert DEFAULT_DATASET_PATH.exists()

    def test_expected_category_is_a_real_category_enum_value(self):
        records = load_ground_truth()

        assert all(isinstance(r.expected_category, Category) for r in records)

    def test_loads_from_an_explicit_path(self, tmp_path):
        data = [
            {
                "message_id": "custom-1",
                "sender": "a@example.com",
                "subject": "Hi",
                "timestamp": "2026-01-01T00:00:00Z",
                "body": "Hello",
                "expected_category": "DIGEST",
                "notes": "test record",
            }
        ]
        path = tmp_path / "custom_dataset.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        records = load_ground_truth(path)

        assert len(records) == 1
        assert records[0].message_id == "custom-1"
        assert records[0].expected_category == Category.DIGEST

    def test_invalid_category_in_dataset_raises(self, tmp_path):
        data = [
            {
                "message_id": "bad-1",
                "sender": "a@example.com",
                "subject": "Hi",
                "timestamp": "2026-01-01T00:00:00Z",
                "body": "Hello",
                "expected_category": "NOT_A_REAL_CATEGORY",
            }
        ]
        path = tmp_path / "bad_dataset.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError):
            load_ground_truth(path)

    def test_missing_required_field_raises(self, tmp_path):
        data = [{"message_id": "bad-2", "expected_category": "DIGEST"}]
        path = tmp_path / "bad_dataset2.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(KeyError):
            load_ground_truth(path)


class TestGroundTruthRecordToEmailMessage:
    def test_converts_to_a_valid_email_message(self):
        record = GroundTruthRecord(
            message_id="msg-1",
            sender="a@example.com",
            subject="Subject",
            body="Body text",
            received_at="2026-01-01T00:00:00+00:00",
            expected_category=Category.NOTIFY,
        )

        email = record.to_email_message()

        assert isinstance(email, EmailMessage)
        assert email.message_id == "msg-1"
        assert email.sender == "a@example.com"
        assert email.subject == "Subject"
        assert email.body == "Body text"

    def test_expected_category_is_not_part_of_email_message(self):
        """EmailMessage has no expected_category field at all — this is
        really just confirming the classifier is never handed the answer.
        """
        record = GroundTruthRecord(
            message_id="msg-1",
            sender="a@example.com",
            received_at="2026-01-01T00:00:00+00:00",
            expected_category=Category.QUARANTINE,
        )

        email = record.to_email_message()

        assert not hasattr(email, "expected_category")
