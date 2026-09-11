"""Tests for app.ui.evaluation_view: pure formatting of EvaluationReport
for display. No streamlit import, no Gemini, no network.
"""

from app.core.decisions import DecisionEngine
from app.core.models import Category, UrgencyLevel
from app.evaluation.dataset import GroundTruthRecord
from app.evaluation.evaluator import evaluate
from app.evaluation.metrics import CategoryMetrics
from app.ui.evaluation_view import (
    category_metrics_rows,
    confusion_matrix_rows,
    coverage_message,
    format_recall,
    mismatch_rows,
)
from app.preferences.manager import PreferenceManager, Preferences
from app.storage.repositories import EmailRepository
from tests.test_models import make_classification, make_email
from datetime import datetime, timezone
import pytest


def gt(message_id, category):
    return GroundTruthRecord(
        message_id=message_id,
        sender="someone@example.com",
        subject="Subject",
        body="",
        received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expected_category=category,
    )


@pytest.fixture
def repo(tmp_path) -> EmailRepository:
    repository = EmailRepository(tmp_path / "view_test.db")
    repository.initialize()
    return repository


@pytest.fixture
def preferences() -> PreferenceManager:
    return PreferenceManager(Preferences())


class TestCoverageMessage:
    def test_full_coverage_message(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        repo.save_classification(email, classification)
        decision = DecisionEngine().decide(email, classification, preferences)
        repo.save_decision(decision)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        message = coverage_message(report)

        assert "Full evaluation" in message
        assert "1" in message

    def test_partial_coverage_message_names_the_counts(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY), gt("msg-2", Category.DIGEST)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        repo.save_classification(email, classification)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        message = coverage_message(report)

        assert "Partial evaluation" in message
        assert "1 of 2" in message
        assert "1 email(s) have not been classified" in message

    def test_never_claims_full_coverage_when_partial(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY), gt("msg-2", Category.DIGEST)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        repo.save_classification(email, classification)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        message = coverage_message(report)

        assert "Full evaluation" not in message


class TestConfusionMatrixRows:
    def test_rows_cover_every_category(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        repo.save_classification(email, classification)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        rows = confusion_matrix_rows(report.classification.confusion_matrix)

        assert len(rows) == 4
        actual_labels = {row["actual \\ predicted"] for row in rows}
        assert actual_labels == {"NOTIFY", "DIGEST", "MUTE", "QUARANTINE"}
        for row in rows:
            assert set(row.keys()) == {"actual \\ predicted", "NOTIFY", "DIGEST", "MUTE", "QUARANTINE"}


class TestCategoryMetricsRows:
    def test_rows_contain_expected_keys(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        repo.save_classification(email, classification)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        rows = category_metrics_rows(report.classification.per_category)

        assert len(rows) == 4
        for row in rows:
            assert set(row.keys()) == {"category", "precision", "recall", "f1", "support"}


class TestFormatRecall:
    def test_zero_support_shows_not_applicable_not_zero_percent(self):
        metrics = CategoryMetrics(category=Category.QUARANTINE, precision=0.0, recall=0.0, f1=0.0, support=0)

        result = format_recall(metrics)

        assert result == "N/A (0 examples)"
        assert "0.0%" not in result

    def test_nonzero_support_shows_percentage(self):
        metrics = CategoryMetrics(category=Category.NOTIFY, precision=1.0, recall=0.833, f1=0.9, support=6)

        result = format_recall(metrics)

        assert result == "83.3%"

    def test_perfect_recall_with_support_shows_100_percent(self):
        metrics = CategoryMetrics(category=Category.QUARANTINE, precision=1.0, recall=1.0, f1=1.0, support=2)

        assert format_recall(metrics) == "100.0%"


class TestMismatchRows:
    def test_no_body_field_ever_present(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY)]
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.DIGEST)  # wrong
        repo.save_classification(email, classification)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)
        rows = mismatch_rows(report.classification_mismatches)

        assert len(rows) == 1
        assert "body" not in rows[0]
        assert rows[0]["expected"] == "NOTIFY"
        assert rows[0]["actual"] == "DIGEST"
