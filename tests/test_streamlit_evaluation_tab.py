"""End-to-end smoke tests for the Evaluation tab in app/streamlit_app.py,
using Streamlit's AppTest harness against a temporary SQLite database.

No Gemini call is possible: DATABASE_PATH is redirected to a temp file,
and app.streamlit_app / app.evaluation never import app.ai.
"""

from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

from app.core.decisions import DecisionEngine
from app.core.models import Category, RiskFlag, UrgencyLevel
from app.preferences.manager import PreferenceManager, Preferences
from app.storage.repositories import EmailRepository
from tests.test_evaluation_evaluator import gt
from tests.test_models import make_classification, make_email

APP_PATH = "app/streamlit_app.py"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "dashboard.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    return path


def seed_full(path, message_id, category, urgency=UrgencyLevel.LOW, risk_flags=None):
    """Stores a classification AND a real decision (via the real
    DecisionEngine) for one message_id."""
    repo = EmailRepository(path)
    repo.initialize()
    preferences = PreferenceManager(Preferences())
    email = make_email(
        message_id=message_id,
        received_at=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc),
    )
    classification = make_classification(
        message_id=message_id, category=category, urgency=urgency, risk_flags=risk_flags or []
    )
    repo.save_classification(email, classification)
    decision = DecisionEngine().decide(email, classification, preferences)
    repo.save_decision(decision)


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    return at


class TestEvaluationTabEmptyDatabase:
    def test_shows_zero_coverage_without_crashing(self, db_path):
        at = run_app()

        assert len(at.exception) == 0
        # With an empty database, the Inbox tab shows its own empty
        # state; the Evaluation tab still renders (0 evaluated, 41 total)
        # rather than crashing or being hidden.
        assert any("Evaluated" == m.label for m in at.metric)


class TestEvaluationTabPartialCoverage:
    def test_reports_partial_coverage_correctly(self, db_path):
        seed_full(db_path, "email-001", Category.NOTIFY, urgency=UrgencyLevel.HIGH)

        at = run_app()

        assert len(at.exception) == 0
        assert any("Partial evaluation" in w.value for w in at.warning)
        values = {m.label: m.value for m in at.metric}
        assert values["Evaluated"] == "1"
        assert values["Total in dataset"] == "41"

    def test_full_coverage_shows_success_not_warning(self, db_path, monkeypatch):
        """Uses a tiny custom ground-truth set (patched dataset loader is
        overkill here); instead we just verify the partial-vs-full
        branching logic end to end using the one email we control,
        against the real 41-record dataset — which will always be
        partial for a single seeded record. To exercise the "full"
        branch, we monkeypatch load_ground_truth to a 1-record set.
        """
        seed_full(db_path, "solo-msg", Category.NOTIFY, urgency=UrgencyLevel.HIGH)

        import app.evaluation.evaluator as evaluator_module

        monkeypatch.setattr(
            evaluator_module,
            "load_ground_truth",
            lambda: [gt("solo-msg", Category.NOTIFY)],
        )

        at = run_app()

        assert len(at.exception) == 0
        assert any("Full evaluation" in s.value for s in at.success)


class TestEvaluationTabMetrics:
    def test_safety_critical_metrics_are_rendered(self, db_path):
        seed_full(db_path, "email-001", Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        seed_full(db_path, "email-033", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])

        at = run_app()

        assert len(at.exception) == 0
        labels = {m.label for m in at.metric}
        assert any("NOTIFY recall" in label for label in labels)
        assert any("QUARANTINE recall" in label for label in labels)

    def test_no_raw_body_appears_in_evaluation_tables(self, db_path):
        seed_full(db_path, "email-001", Category.NOTIFY, urgency=UrgencyLevel.HIGH)

        at = run_app()

        assert len(at.exception) == 0
        for df in at.dataframe:
            columns = {str(c).lower() for c in df.value.columns}
            assert "body" not in columns


class TestEvaluationTabDoesNotBreakInboxTab:
    def test_inbox_tab_still_works_alongside_evaluation_tab(self, db_path):
        seed_full(db_path, "email-001", Category.NOTIFY, urgency=UrgencyLevel.HIGH)

        at = run_app()

        assert len(at.exception) == 0
        values = {m.label: m.value for m in at.metric}
        assert values["Total processed"] == "1"
        assert values["🔴 NOTIFY"] == "1"
