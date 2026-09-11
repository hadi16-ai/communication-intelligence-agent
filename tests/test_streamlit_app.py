"""End-to-end smoke tests for app/streamlit_app.py using Streamlit's own
AppTest harness — runs the real script (no mocked UI internals) against a
temporary SQLite database. No Gemini call is possible here: the dashboard
module never imports app.ai, and DATABASE_PATH is redirected to a
temporary file before the app runs.
"""

from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

from app.core.models import Category, RiskFlag, UrgencyLevel
from app.storage.repositories import EmailRepository
from tests.test_models import make_classification, make_decision, make_email

APP_PATH = "app/streamlit_app.py"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "dashboard.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    return path


def seed(path, message_id, category, urgency=UrgencyLevel.LOW, risk_flags=None, sender=None, subject=None):
    repo = EmailRepository(path)
    repo.initialize()
    email = make_email(
        message_id=message_id,
        sender=sender or "someone@example.com",
        subject=subject or "A subject",
        received_at=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc),
    )
    classification = make_classification(
        message_id=message_id, category=category, urgency=urgency, risk_flags=risk_flags or []
    )
    repo.save_classification(email, classification)
    decision = make_decision(classification=classification, category=category)
    repo.save_decision(decision)


class TestEmptyState:
    def test_empty_database_shows_info_message_and_no_exceptions(self, db_path):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.exception) == 0
        assert any("No processed emails yet" in info.value for info in at.info)
        # The Inbox tab itself shows no metrics when empty; the sibling
        # Evaluation tab (added in M9) always renders its own overview
        # metrics (0/41, 0% coverage, etc.), so assert on the absence of
        # Inbox-specific metrics rather than zero metrics globally.
        inbox_labels = {"🔴 NOTIFY", "🟡 DIGEST", "⚪ MUTE", "🛡️ QUARANTINE", "Total processed"}
        assert not any(m.label in inbox_labels for m in at.metric)


class TestPopulatedDashboard:
    def test_runs_without_exceptions(self, db_path):
        seed(db_path, "msg-1", Category.NOTIFY)
        seed(db_path, "msg-2", Category.DIGEST)

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.exception) == 0

    def test_summary_metrics_reflect_stored_counts(self, db_path):
        seed(db_path, "msg-1", Category.NOTIFY)
        seed(db_path, "msg-2", Category.NOTIFY)
        seed(db_path, "msg-3", Category.DIGEST)
        seed(db_path, "msg-4", Category.MUTE)
        seed(db_path, "msg-5", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        values = {m.label: m.value for m in at.metric}
        assert values["🔴 NOTIFY"] == "2"
        assert values["🟡 DIGEST"] == "1"
        assert values["⚪ MUTE"] == "1"
        assert values["🛡️ QUARANTINE"] == "1"
        assert values["Total processed"] == "5"

    def test_table_shows_expected_row_count(self, db_path):
        seed(db_path, "msg-1", Category.NOTIFY)
        seed(db_path, "msg-2", Category.DIGEST)

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.dataframe) == 1
        assert len(at.dataframe[0].value) == 2

    def test_filtering_by_category_reduces_table(self, db_path):
        seed(db_path, "msg-1", Category.NOTIFY)
        seed(db_path, "msg-2", Category.DIGEST)
        seed(db_path, "msg-3", Category.NOTIFY)

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        at.radio[0].set_value("🔴 NOTIFY").run()

        assert len(at.dataframe[0].value) == 2

    def test_search_reduces_table(self, db_path):
        seed(db_path, "msg-1", Category.NOTIFY, subject="Interview availability")
        seed(db_path, "msg-2", Category.NOTIFY, subject="Totally unrelated")

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        at.text_input[0].set_value("interview").run()

        assert len(at.dataframe[0].value) == 1

    def test_quarantine_record_shows_error_banner(self, db_path):
        seed(db_path, "msg-1", Category.QUARANTINE, risk_flags=[RiskFlag.PHISHING])

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.error) >= 1
        assert any("QUARANTINE" in e.value for e in at.error)

    def test_no_raw_body_ever_appears_anywhere_on_the_page(self, db_path):
        """EmailMessage.body is never stored in SQLite in the first place,
        so this is really asserting the dashboard doesn't leak anything
        resembling body content into any rendered widget.
        """
        seed(db_path, "msg-1", Category.NOTIFY, subject="Interview availability")

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.exception) == 0
        for df in at.dataframe:
            columns = {str(c).lower() for c in df.value.columns}
            assert "body" not in columns

    def test_undecided_record_does_not_crash_dashboard(self, db_path):
        repo = EmailRepository(db_path)
        repo.initialize()
        email = make_email(message_id="msg-1")
        classification = make_classification(message_id="msg-1", category=Category.DIGEST)
        repo.save_classification(email, classification)  # no decision saved

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()

        assert len(at.exception) == 0
        values = {m.label: m.value for m in at.metric}
        assert values["Total processed"] == "1"
        assert values["🟡 DIGEST"] == "0"
