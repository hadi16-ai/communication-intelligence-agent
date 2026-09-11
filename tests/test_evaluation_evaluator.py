"""Tests for app.evaluation.evaluator: the orchestration layer that
combines ground truth, stored predictions, and the real DecisionEngine
into an EvaluationReport.

No Gemini/network calls anywhere — classifications are inserted directly
into a temporary SQLite database exactly as GeminiClassifier's output
would be, and decisions are produced by the real, unmodified
DecisionEngine given a deterministic ground-truth stand-in.
"""

from datetime import datetime, timezone

import pytest

from app.core.decisions import DecisionEngine
from app.core.models import Category, RiskFlag, UrgencyLevel
from app.evaluation.dataset import GroundTruthRecord
from app.evaluation.evaluator import (
    derive_expected_decision,
    derive_ground_truth_classification,
    evaluate,
)
from app.preferences.manager import PreferenceManager, Preferences, RuleSet
from app.storage.repositories import EmailRepository
from tests.test_models import make_classification, make_email


def gt(message_id, category, sender="someone@example.com", subject="A subject", notes=""):
    return GroundTruthRecord(
        message_id=message_id,
        sender=sender,
        subject=subject,
        body="",
        received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expected_category=category,
        notes=notes,
    )


@pytest.fixture
def repo(tmp_path) -> EmailRepository:
    repository = EmailRepository(tmp_path / "eval_test.db")
    repository.initialize()
    return repository


@pytest.fixture
def preferences() -> PreferenceManager:
    return PreferenceManager(Preferences())  # no rules; pure fallback behavior


def store_classification(repo, message_id, category, urgency=UrgencyLevel.LOW, risk_flags=None, sender="someone@example.com", subject="A subject"):
    email = make_email(message_id=message_id, sender=sender, subject=subject)
    classification = make_classification(
        message_id=message_id, category=category, urgency=urgency, risk_flags=risk_flags or []
    )
    repo.save_classification(email, classification)
    return email, classification


def store_decision(repo, email, classification, preferences):
    decision = DecisionEngine().decide(email, classification, preferences)
    repo.save_decision(decision)
    return decision


class TestDeriveGroundTruthClassification:
    def test_quarantine_gets_a_risk_flag(self):
        email = make_email(message_id="msg-1")
        classification = derive_ground_truth_classification(email, Category.QUARANTINE)

        assert classification.category == Category.QUARANTINE
        assert RiskFlag.OTHER in classification.risk_flags

    def test_notify_gets_high_urgency(self):
        email = make_email(message_id="msg-1")
        classification = derive_ground_truth_classification(email, Category.NOTIFY)

        assert classification.urgency == UrgencyLevel.HIGH
        assert classification.risk_flags == []

    def test_message_id_matches_the_email(self):
        email = make_email(message_id="msg-42")
        classification = derive_ground_truth_classification(email, Category.DIGEST)

        assert classification.message_id == "msg-42"


class TestDeriveExpectedDecision:
    def test_uses_the_real_decision_engine(self, preferences):
        """With no preference rules configured, a NOTIFY-expected record
        (which derives HIGH urgency) should fall through to the real
        DecisionEngine's fallback: HIGH urgency escalates to NOTIFY.
        """
        email = make_email(message_id="msg-1")
        decision = derive_expected_decision(email, Category.NOTIFY, preferences, DecisionEngine())

        assert decision.category == Category.NOTIFY

    def test_quarantine_always_wins_regardless_of_preferences(self, preferences):
        email = make_email(message_id="msg-1")
        decision = derive_expected_decision(email, Category.QUARANTINE, preferences, DecisionEngine())

        assert decision.category == Category.QUARANTINE


class TestPartialEvaluation:
    def test_evaluates_only_records_with_stored_classifications(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY), gt("msg-2", Category.DIGEST), gt("msg-3", Category.MUTE)]
        email, classification = store_classification(repo, "msg-1", Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        store_decision(repo, email, classification, preferences)
        # msg-2 and msg-3 are never classified — must not be fabricated.

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.coverage.evaluated_count == 1
        assert report.coverage.total_count == 3
        assert report.coverage.missing_count == 2
        assert set(report.coverage.missing_message_ids) == {"msg-2", "msg-3"}
        assert report.coverage.coverage_percent == pytest.approx(100 / 3)
        assert report.coverage.is_partial is True
        assert report.classification.total == 1

    def test_full_coverage_is_not_partial(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY)]
        email, classification = store_classification(repo, "msg-1", Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        store_decision(repo, email, classification, preferences)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.coverage.is_partial is False
        assert report.coverage.coverage_percent == 100.0

    def test_zero_evaluated_gives_zero_coverage_not_an_error(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY), gt("msg-2", Category.DIGEST)]

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.coverage.evaluated_count == 0
        assert report.coverage.coverage_percent == 0.0
        assert report.classification.total == 0
        assert report.classification.accuracy == 0.0

    def test_never_calls_a_classifier(self, repo, preferences, monkeypatch):
        """There is no classifier dependency at all in evaluate() — this
        just documents/guards that fact by making sure get_classifier
        would blow up if it were ever (wrongly) invoked.
        """
        ground_truth = [gt("msg-1", Category.NOTIFY)]

        def fail_if_called(*args, **kwargs):
            raise AssertionError("evaluate() must never call a classifier")

        monkeypatch.setattr("app.ai.classifier.get_classifier", fail_if_called)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.coverage.evaluated_count == 0


class TestMissingDecisionButPresentClassification:
    def test_classification_evaluated_even_without_a_decision(self, repo, preferences):
        store_classification(repo, "msg-1", Category.NOTIFY)  # no decision saved
        ground_truth = [gt("msg-1", Category.NOTIFY)]

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.classification.total == 1
        assert report.decision.total == 0  # nothing to compare on the decision axis
        assert report.coverage.evaluated_count == 1  # still counts as "evaluated"


class TestDuplicateMessageIds:
    def test_duplicate_ground_truth_ids_are_deduplicated_last_wins(self, repo, preferences):
        ground_truth = [
            gt("msg-1", Category.NOTIFY, notes="first"),
            gt("msg-1", Category.DIGEST, notes="second, should win"),
        ]
        store_classification(repo, "msg-1", Category.DIGEST)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.coverage.total_count == 1
        assert report.classification.total == 1
        assert report.classification.accuracy == 1.0  # DIGEST expected == DIGEST actual


class TestClassificationMismatches:
    def test_mismatches_are_reported_with_expected_and_actual(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.NOTIFY, notes="should have been notify")]
        store_classification(repo, "msg-1", Category.DIGEST)  # wrong

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert len(report.classification_mismatches) == 1
        mismatch = report.classification_mismatches[0]
        assert mismatch.message_id == "msg-1"
        assert mismatch.expected == Category.NOTIFY
        assert mismatch.actual == Category.DIGEST
        assert mismatch.notes == "should have been notify"

    def test_no_mismatches_when_everything_matches(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.DIGEST)]
        store_classification(repo, "msg-1", Category.DIGEST)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.classification_mismatches == ()


class TestNotifyAndQuarantineRecall:
    def test_notify_recall_reflects_missed_notify_emails(self, repo, preferences):
        ground_truth = [
            gt("msg-1", Category.NOTIFY),
            gt("msg-2", Category.NOTIFY),
        ]
        store_classification(repo, "msg-1", Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        store_classification(repo, "msg-2", Category.DIGEST)  # missed a real NOTIFY

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.notify_recall_classification == pytest.approx(0.5)

    def test_quarantine_recall_is_perfect_when_all_quarantine_caught(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.QUARANTINE), gt("msg-2", Category.QUARANTINE)]
        store_classification(repo, "msg-1", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])
        store_classification(repo, "msg-2", Category.QUARANTINE, risk_flags=[RiskFlag.SCAM])

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.quarantine_recall_classification == 1.0

    def test_quarantine_recall_drops_when_a_quarantine_email_is_missed(self, repo, preferences):
        ground_truth = [gt("msg-1", Category.QUARANTINE), gt("msg-2", Category.QUARANTINE)]
        store_classification(repo, "msg-1", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])
        store_classification(repo, "msg-2", Category.NOTIFY)  # a QUARANTINE email slipped through

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        assert report.quarantine_recall_classification == pytest.approx(0.5)

    def test_decision_level_notify_recall_can_differ_from_classification_level(self, repo):
        """Personalization can change the picture: even if the AI
        correctly classified something as NOTIFY, a preference rule could
        (in principle) still change the final decision — classification
        and decision recall are genuinely different questions.
        """
        preferences = PreferenceManager(Preferences(mute_rules=RuleSet(keywords=["a subject"])))
        ground_truth = [gt("msg-1", Category.NOTIFY, subject="a subject")]
        email, classification = store_classification(
            repo, "msg-1", Category.NOTIFY, urgency=UrgencyLevel.LOW, subject="a subject"
        )
        # Real pipeline decision, using the ACTUAL classification (LOW
        # urgency, so the mute rule wins) — simulating what M7 would have
        # stored, distinct from the ground-truth-derived expectation.
        decision = DecisionEngine().decide(email, classification, preferences)
        repo.save_decision(decision)

        report = evaluate(repo, preferences, DecisionEngine(), ground_truth=ground_truth)

        # Classification was correct (NOTIFY == NOTIFY)...
        assert report.notify_recall_classification == 1.0
        # ...but the actual decision was muted due to the low-urgency
        # actual classification, while the expected-decision derivation
        # uses HIGH urgency for a NOTIFY ground truth, expecting NOTIFY.
        assert decision.category == Category.MUTE
        assert report.notify_recall_decision == 0.0


class TestArchitecturalIsolation:
    def test_evaluator_module_never_imports_gemini_or_genai(self):
        import app.evaluation.evaluator as evaluator_module

        assert "genai" not in evaluator_module.__dict__
        assert "GeminiClassifier" not in dir(evaluator_module)
        assert "get_classifier" not in dir(evaluator_module)


class TestDatasetCompatibility:
    def test_evaluate_runs_against_the_real_shipped_dataset(self, repo, preferences):
        """Uses the full, real data/eval_dataset.json (via the default
        `ground_truth=None`) with only a couple of stored classifications
        — confirming the evaluator handles the real 41-record dataset
        shape without needing all of it processed.
        """
        store_classification(repo, "email-001", Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        store_classification(repo, "email-033", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])

        report = evaluate(repo, preferences, DecisionEngine())

        assert report.coverage.total_count == 41
        assert report.coverage.evaluated_count == 2
        assert report.coverage.is_partial is True
