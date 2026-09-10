"""Tests for app.ui.dashboard_data: pure, streamlit-free logic backing the
dashboard. No Gemini, no network, temporary SQLite databases only.
"""

import pytest

from app.core.decisions import DecisionEngine
from app.core.models import Category, DecisionSource, RiskFlag, UrgencyLevel
from app.storage.repositories import EmailRepository
from app.ui.dashboard_data import (
    decision_status_label,
    decision_trace_label,
    filter_records,
    format_risk_flags,
    load_records,
    summarize,
    to_table_row,
)
from tests.test_models import make_classification, make_decision, make_email


@pytest.fixture
def repo(tmp_path) -> EmailRepository:
    repository = EmailRepository(tmp_path / "dashboard_test.db")
    repository.initialize()
    return repository


def seed(repo, message_id, category, urgency=UrgencyLevel.LOW, risk_flags=None, sender=None, subject=None, with_decision=True):
    email = make_email(
        message_id=message_id,
        sender=sender or "someone@example.com",
        subject=subject or "A subject",
    )
    classification = make_classification(
        message_id=message_id,
        category=category,
        urgency=urgency,
        risk_flags=risk_flags or [],
    )
    repo.save_classification(email, classification)
    if with_decision:
        decision = make_decision(classification=classification, category=category)
        repo.save_decision(decision)
    return email, classification


class TestLoadRecords:
    def test_empty_database_returns_empty_list(self, repo):
        assert load_records(repo) == []

    def test_returns_all_saved_records(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)
        seed(repo, "msg-2", Category.DIGEST)

        records = load_records(repo)

        assert {r.message_id for r in records} == {"msg-1", "msg-2"}


class TestSummarize:
    def test_empty_list_gives_all_zero_counts(self):
        summary = summarize([])
        assert (summary.notify, summary.digest, summary.mute, summary.quarantine, summary.total) == (0, 0, 0, 0, 0)
        assert summary.pending == 0

    def test_counts_by_final_decision_category(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)
        seed(repo, "msg-2", Category.NOTIFY)
        seed(repo, "msg-3", Category.DIGEST)
        seed(repo, "msg-4", Category.MUTE)
        seed(repo, "msg-5", Category.QUARANTINE)

        summary = summarize(load_records(repo))

        assert summary.notify == 2
        assert summary.digest == 1
        assert summary.mute == 1
        assert summary.quarantine == 1
        assert summary.total == 5

    def test_counts_use_decision_category_not_classification_category(self, repo):
        """A record whose AI classification suggested DIGEST but whose
        final decision was overridden to NOTIFY must be counted as
        NOTIFY, not DIGEST — the decision is what "actually happened."
        """
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.DIGEST)
        repo.save_classification(email, classification)
        decision = make_decision(
            classification=classification, category=Category.NOTIFY, source=DecisionSource.PREFERENCE_OVERRIDE
        )
        repo.save_decision(decision)

        summary = summarize(load_records(repo))

        assert summary.notify == 1
        assert summary.digest == 0

    def test_classified_but_not_decided_record_counts_in_total_only(self, repo):
        seed(repo, "msg-1", Category.NOTIFY, with_decision=False)

        summary = summarize(load_records(repo))

        assert summary.total == 1
        assert summary.pending == 1
        assert summary.notify == 0


class TestFilterRecords:
    def test_no_filter_returns_everything(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)
        seed(repo, "msg-2", Category.MUTE)

        assert len(filter_records(load_records(repo))) == 2

    def test_filter_by_category(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)
        seed(repo, "msg-2", Category.MUTE)
        seed(repo, "msg-3", Category.NOTIFY)

        result = filter_records(load_records(repo), category=Category.NOTIFY)

        assert {r.message_id for r in result} == {"msg-1", "msg-3"}

    def test_filter_by_category_excludes_undecided_records(self, repo):
        seed(repo, "msg-1", Category.NOTIFY, with_decision=False)

        result = filter_records(load_records(repo), category=Category.NOTIFY)

        assert result == []

    def test_search_by_sender_case_insensitive(self, repo):
        seed(repo, "msg-1", Category.DIGEST, sender="Recruiter@BrightPath.example.com")
        seed(repo, "msg-2", Category.DIGEST, sender="friend@example.com")

        result = filter_records(load_records(repo), search="brightpath")

        assert [r.message_id for r in result] == ["msg-1"]

    def test_search_by_subject(self, repo):
        seed(repo, "msg-1", Category.DIGEST, subject="Interview availability")
        seed(repo, "msg-2", Category.DIGEST, subject="Weekend plans")

        result = filter_records(load_records(repo), search="interview")

        assert [r.message_id for r in result] == ["msg-1"]

    def test_category_and_search_combine(self, repo):
        seed(repo, "msg-1", Category.NOTIFY, subject="Interview availability")
        seed(repo, "msg-2", Category.MUTE, subject="Interview availability")

        result = filter_records(load_records(repo), category=Category.NOTIFY, search="interview")

        assert [r.message_id for r in result] == ["msg-1"]

    def test_blank_search_does_not_filter(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)

        result = filter_records(load_records(repo), search="   ")

        assert len(result) == 1


class TestFormatRiskFlags:
    def test_empty_list_renders_as_dash(self):
        assert format_risk_flags([]) == "—"

    def test_flags_are_comma_joined(self):
        assert format_risk_flags([RiskFlag.PHISHING, RiskFlag.SCAM]) == "PHISHING, SCAM"


class TestDecisionStatusLabel:
    def test_undecided_record_shows_pending(self, repo):
        seed(repo, "msg-1", Category.NOTIFY, with_decision=False)
        record = load_records(repo)[0]

        assert decision_status_label(record) == "PENDING"

    def test_decided_record_shows_category(self, repo):
        seed(repo, "msg-1", Category.QUARANTINE)
        record = load_records(repo)[0]

        assert decision_status_label(record) == "QUARANTINE"


class TestDecisionTraceLabel:
    def test_preference_override_label(self):
        classification = make_classification(category=Category.DIGEST)
        decision = make_decision(
            classification=classification, category=Category.NOTIFY, source=DecisionSource.PREFERENCE_OVERRIDE
        )
        assert "preference rule" in decision_trace_label(decision).lower()

    def test_ai_classification_label(self):
        classification = make_classification(category=Category.DIGEST)
        decision = make_decision(
            classification=classification, category=Category.DIGEST, source=DecisionSource.AI_CLASSIFICATION
        )
        assert "classification" in decision_trace_label(decision).lower()


class TestToTableRow:
    def test_never_includes_raw_body(self, repo):
        seed(repo, "msg-1", Category.NOTIFY)
        record = load_records(repo)[0]

        row = to_table_row(record)

        assert "body" not in row
        assert all("body" not in str(key).lower() for key in row.keys())

    def test_row_contains_expected_fields(self, repo):
        seed(repo, "msg-1", Category.NOTIFY, urgency=UrgencyLevel.HIGH, risk_flags=[RiskFlag.PHISHING])
        record = load_records(repo)[0]

        row = to_table_row(record)

        assert row["message_id"] == "msg-1"
        assert row["decision"] == "NOTIFY"
        assert row["urgency"] == "HIGH"
        assert row["risk_flags"] == "PHISHING"
        assert 0.0 <= row["confidence"] <= 1.0


class TestRealDecisionEngineIntegration:
    def test_summary_reflects_a_real_decision_engine_run(self, repo):
        """Sanity check using the actual DecisionEngine + a protected/urgent
        email, rather than only the make_decision() test helper, to make
        sure summarize() works against genuinely produced Decisions too.
        """
        from app.preferences.manager import PreferenceManager, Preferences, RuleSet

        preferences = PreferenceManager(
            Preferences(quarantine_rules=RuleSet(keywords=["gift card"]))
        )
        email = make_email(message_id="msg-real", subject="Quick favor", body="Please send a gift card now.")
        classification = make_classification(message_id="msg-real", category=Category.DIGEST, risk_flags=[])
        decision = DecisionEngine().decide(email, classification, preferences)

        repo.save_classification(email, classification)
        repo.save_decision(decision)

        summary = summarize(load_records(repo))

        assert summary.quarantine == 1
