"""Tests for app.storage.repositories.EmailRepository: persistence,
classification caching, and decision storage.

Every test uses a fresh temporary SQLite database (via the `repo` fixture)
— never the production database. No Gemini API calls, no network, and no
GEMINI_API_KEY are required anywhere in this file.
"""

import sqlite3

import pytest

from app.core.models import Category, DecisionSource, RiskFlag, UrgencyLevel
from app.storage.database import get_connection
from app.storage.repositories import EmailRepository, StorageError
from tests.test_models import make_classification, make_decision, make_email


@pytest.fixture
def repo(tmp_path) -> EmailRepository:
    repository = EmailRepository(tmp_path / "test.db")
    repository.initialize()
    return repository


class TestSaveAndRetrieve:
    def test_save_and_get_record_round_trip(self, repo):
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", risk_flags=[RiskFlag.PHISHING])

        repo.save_classification(email, classification)
        record = repo.get_record("msg-1")

        assert record is not None
        assert record.message_id == "msg-1"
        assert record.sender == email.sender
        assert record.subject == email.subject
        assert record.classification == classification
        assert record.decision is None

    def test_missing_message_id_returns_none(self, repo):
        assert repo.get_record("does-not-exist") is None
        assert repo.get_classification("does-not-exist") is None
        assert repo.get_decision("does-not-exist") is None
        assert repo.has_classification("does-not-exist") is False

    def test_duplicate_message_id_does_not_create_duplicate_rows(self, repo):
        email = make_email(message_id="msg-1")
        classification_1 = make_classification(message_id="msg-1", summary="first summary version")
        classification_2 = make_classification(message_id="msg-1", summary="second summary version")

        repo.save_classification(email, classification_1)
        repo.save_classification(email, classification_2)

        with get_connection(repo._db_path) as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM email_records").fetchone()["c"]
        assert count == 1

        stored = repo.get_classification("msg-1")
        assert stored.summary == "second summary version"

    def test_classification_email_message_id_mismatch_rejected(self, repo):
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-DIFFERENT")

        with pytest.raises(ValueError):
            repo.save_classification(email, classification)

    def test_save_decision_requires_existing_classification(self, repo):
        classification = make_classification(message_id="msg-never-saved")
        decision = make_decision(classification=classification)

        with pytest.raises(ValueError):
            repo.save_decision(decision)

    def test_save_decision_after_classification(self, repo):
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1")
        decision = make_decision(
            classification=classification,
            category=Category.NOTIFY,
            source=DecisionSource.PREFERENCE_OVERRIDE,
        )

        repo.save_classification(email, classification)
        repo.save_decision(decision)

        stored_decision = repo.get_decision("msg-1")
        assert stored_decision is not None
        assert stored_decision.category == Category.NOTIFY
        assert stored_decision.source == DecisionSource.PREFERENCE_OVERRIDE
        assert stored_decision.reasoning == decision.reasoning
        assert stored_decision.classification == classification

        record = repo.get_record("msg-1")
        assert record.decision == stored_decision

    def test_reclassifying_does_not_erase_an_existing_decision(self, repo):
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.DIGEST)
        decision = make_decision(classification=classification, category=Category.NOTIFY)
        repo.save_classification(email, classification)
        repo.save_decision(decision)

        reclassification = make_classification(message_id="msg-1", category=Category.DIGEST, summary="updated")
        repo.save_classification(email, reclassification)

        assert repo.get_decision("msg-1") is not None
        assert repo.get_decision("msg-1").category == Category.NOTIFY


class TestClassificationRoundTrip:
    def test_all_fields_survive_round_trip(self, repo):
        email = make_email(message_id="msg-rt")
        classification = make_classification(
            message_id="msg-rt",
            category=Category.QUARANTINE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.7654,
            risk_flags=[RiskFlag.PHISHING, RiskFlag.SUSPICIOUS_LINK],
            summary="A phishing summary.",
            reasoning="Detailed reasoning text.",
        )

        repo.save_classification(email, classification)
        restored = repo.get_classification("msg-rt")

        assert restored == classification

    def test_empty_risk_flags_round_trip(self, repo):
        email = make_email(message_id="msg-empty-risk")
        classification = make_classification(message_id="msg-empty-risk", risk_flags=[])

        repo.save_classification(email, classification)
        restored = repo.get_classification("msg-empty-risk")

        assert restored.risk_flags == []


class TestDecisionRoundTrip:
    def test_decision_fields_survive_round_trip(self, repo):
        email = make_email(message_id="msg-d1")
        classification = make_classification(message_id="msg-d1")
        decision = make_decision(
            classification=classification,
            category=Category.MUTE,
            reasoning="Matches mute preferences.",
            source=DecisionSource.PREFERENCE_OVERRIDE,
        )

        repo.save_classification(email, classification)
        repo.save_decision(decision)
        restored = repo.get_decision("msg-d1")

        assert restored == decision


class TestCaching:
    def test_first_lookup_is_a_cache_miss(self, repo):
        assert repo.has_classification("msg-new") is False
        assert repo.get_classification("msg-new") is None

    def test_lookup_becomes_a_cache_hit_after_saving(self, repo):
        email = make_email(message_id="msg-cache")
        classification = make_classification(message_id="msg-cache")

        assert repo.has_classification("msg-cache") is False
        repo.save_classification(email, classification)
        assert repo.has_classification("msg-cache") is True

    def test_cached_classification_exactly_matches_original(self, repo):
        email = make_email(message_id="msg-cache-2")
        classification = make_classification(
            message_id="msg-cache-2",
            category=Category.NOTIFY,
            urgency=UrgencyLevel.HIGH,
            confidence=0.88,
            risk_flags=[RiskFlag.OTHER],
            summary="summary text",
            reasoning="reasoning text",
        )
        repo.save_classification(email, classification)

        assert repo.get_classification("msg-cache-2") == classification

    def test_repeated_lookup_does_not_require_gemini(self, repo, monkeypatch):
        """Mirrors how M7's pipeline will use this repository: if
        has_classification() is True, the classifier must never be
        invoked. Guards the intended cache-first usage pattern.
        """
        email = make_email(message_id="msg-no-gemini")
        classification = make_classification(message_id="msg-no-gemini")
        repo.save_classification(email, classification)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Gemini must not be called when a cached classification exists")

        monkeypatch.setattr("app.ai.classifier.get_classifier", fail_if_called)

        if repo.has_classification(email.message_id):
            from app.ai import classifier as classifier_module  # noqa: F401  (ensure patch target exists)

            result = repo.get_classification(email.message_id)
        else:
            from app.ai.classifier import get_classifier

            result = get_classifier().classify(email)

        assert result == classification


class TestIntegrityAndCorruption:
    def test_malformed_risk_flags_json_raises_storage_error(self, repo):
        email = make_email(message_id="msg-corrupt")
        classification = make_classification(message_id="msg-corrupt")
        repo.save_classification(email, classification)

        with get_connection(repo._db_path) as conn:
            conn.execute(
                "UPDATE email_records SET classification_risk_flags = ? WHERE message_id = ?",
                ("not valid json{{{", "msg-corrupt"),
            )
            conn.commit()

        with pytest.raises(StorageError):
            repo.get_classification("msg-corrupt")

    def test_invalid_category_value_raises_storage_error(self, repo):
        email = make_email(message_id="msg-bad-category")
        classification = make_classification(message_id="msg-bad-category")
        repo.save_classification(email, classification)

        with get_connection(repo._db_path) as conn:
            conn.execute(
                "UPDATE email_records SET classification_category = ? WHERE message_id = ?",
                ("NOT_A_REAL_CATEGORY", "msg-bad-category"),
            )
            conn.commit()

        with pytest.raises(StorageError):
            repo.get_classification("msg-bad-category")

    def test_invalid_stored_decision_raises_storage_error(self, repo):
        email = make_email(message_id="msg-bad-decision")
        classification = make_classification(message_id="msg-bad-decision")
        decision = make_decision(classification=classification)
        repo.save_classification(email, classification)
        repo.save_decision(decision)

        with get_connection(repo._db_path) as conn:
            conn.execute(
                "UPDATE email_records SET decision_source = ? WHERE message_id = ?",
                ("NOT_A_REAL_SOURCE", "msg-bad-decision"),
            )
            conn.commit()

        with pytest.raises(StorageError):
            repo.get_decision("msg-bad-decision")

    def test_message_id_is_consistent_across_email_classification_and_decision(self, repo):
        email = make_email(message_id="msg-consistent")
        classification = make_classification(message_id="msg-consistent")
        decision = make_decision(classification=classification)

        repo.save_classification(email, classification)
        repo.save_decision(decision)

        record = repo.get_record("msg-consistent")

        assert record.message_id == classification.message_id == decision.message_id

    def test_no_raw_email_body_column_in_schema(self, repo):
        with get_connection(repo._db_path) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}

        assert "body" not in columns
        assert "raw_body" not in columns
        assert "content" not in columns

    def test_sql_special_characters_are_stored_literally_not_executed(self, repo):
        """A subject/message_id containing SQL-special characters must be
        stored and retrieved literally — proof the queries are
        parameterized rather than built by string interpolation.
        """
        malicious_subject = "'; DROP TABLE email_records; --"
        email = make_email(message_id="msg-injection", subject=malicious_subject)
        classification = make_classification(message_id="msg-injection")

        repo.save_classification(email, classification)
        record = repo.get_record("msg-injection")

        assert record.subject == malicious_subject
        with get_connection(repo._db_path) as conn:
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "email_records" in tables
