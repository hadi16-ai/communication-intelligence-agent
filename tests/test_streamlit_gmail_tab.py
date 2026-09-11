"""End-to-end smoke tests for the Gmail tab in app/streamlit_app.py, using
Streamlit's AppTest harness.

No real Gmail API calls and no Gemini calls anywhere: DATABASE_PATH and
GMAIL_TOKEN_PATH point at temp paths, and the Gmail-client/classifier
construction functions are monkeypatched to fakes before any button click.
"""

import pytest
from streamlit.testing.v1 import AppTest

from app.core.models import Category, UrgencyLevel
from tests.fakes import FakeClassifier
from tests.test_gmail_client import FakeGmailClient, valid_message
from tests.test_models import make_classification

APP_PATH = "app/streamlit_app.py"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "dashboard.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    return path


@pytest.fixture
def token_path(tmp_path, monkeypatch):
    path = tmp_path / "credentials" / "token.json"
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(path))
    return path


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    return at


def connect(token_path):
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("{}", encoding="utf-8")


class TestNotConnected:
    def test_shows_connect_instructions_when_no_token(self, db_path, token_path):
        at = run_app()

        assert len(at.exception) == 0
        assert any("not connected" in w.value.lower() for w in at.warning)
        assert any("authorize_gmail.py" in c.value for c in at.code)

    def test_no_fetch_button_when_not_connected(self, db_path, token_path):
        at = run_app()

        assert len(at.button) == 0

    def test_readonly_notice_always_shown(self, db_path, token_path):
        at = run_app()

        assert any("Read-only Gmail access" in i.value for i in at.info)


class TestConnected:
    def test_shows_connected_status_and_fetch_controls(self, db_path, token_path):
        connect(token_path)

        at = run_app()

        assert len(at.exception) == 0
        assert any("connected" in s.value.lower() for s in at.success)
        assert len(at.number_input) >= 1
        assert any("Fetch and process" in b.label for b in at.button)


class TestFetchAndProcess:
    def test_clicking_fetch_processes_messages_through_the_real_pipeline(self, db_path, token_path, monkeypatch):
        connect(token_path)
        fake_gmail_client = FakeGmailClient(
            ids=["g1", "g2"],
            messages={"g1": valid_message("g1", "Hello"), "g2": valid_message("g2", "World")},
        )
        fake_classifier = FakeClassifier(
            default=make_classification(category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        )
        monkeypatch.setattr("app.gmail.client.GmailClient.from_token_file", lambda *a, **kw: fake_gmail_client)
        monkeypatch.setattr("app.ai.classifier.get_classifier", lambda settings: fake_classifier)

        at = run_app()
        at.button[0].click().run()

        assert len(at.exception) == 0
        assert any("Fetched 2 message(s)" in s.value for s in at.success)

    def test_processed_messages_appear_in_the_inbox_tab(self, db_path, token_path, monkeypatch):
        connect(token_path)
        fake_gmail_client = FakeGmailClient(ids=["g1"], messages={"g1": valid_message("g1", "Hello")})
        fake_classifier = FakeClassifier(
            default=make_classification(category=Category.NOTIFY, urgency=UrgencyLevel.HIGH)
        )
        monkeypatch.setattr("app.gmail.client.GmailClient.from_token_file", lambda *a, **kw: fake_gmail_client)
        monkeypatch.setattr("app.ai.classifier.get_classifier", lambda settings: fake_classifier)

        at = run_app()
        at.button[0].click().run()

        values = {m.label: m.value for m in at.metric}
        assert values["Total processed"] == "1"
        assert values["🔴 NOTIFY"] == "1"

    def test_gmail_auth_error_shows_error_not_crash(self, db_path, token_path, monkeypatch):
        connect(token_path)
        from app.gmail.client import GmailAuthError

        def raise_auth_error(*args, **kwargs):
            raise GmailAuthError("simulated auth failure")

        monkeypatch.setattr("app.gmail.client.GmailClient.from_token_file", raise_auth_error)

        at = run_app()
        at.button[0].click().run()

        assert len(at.exception) == 0
        assert any("Gmail authentication error" in e.value for e in at.error)

    def test_gmail_api_error_shows_error_not_crash(self, db_path, token_path, monkeypatch):
        connect(token_path)
        from app.gmail.client import GmailAPIError

        fake_gmail_client = FakeGmailClient(ids=[])

        def raise_api_error(max_results):
            raise GmailAPIError("simulated API failure")

        fake_gmail_client.list_message_ids = raise_api_error
        monkeypatch.setattr("app.gmail.client.GmailClient.from_token_file", lambda *a, **kw: fake_gmail_client)
        monkeypatch.setattr(
            "app.ai.classifier.get_classifier",
            lambda settings: FakeClassifier(default=make_classification()),
        )

        at = run_app()
        at.button[0].click().run()

        assert len(at.exception) == 0
        assert any("Gmail API error" in e.value for e in at.error)

    def test_second_fetch_of_same_messages_uses_cache_not_the_classifier_again(
        self, db_path, token_path, monkeypatch
    ):
        connect(token_path)
        fake_gmail_client = FakeGmailClient(ids=["g1"], messages={"g1": valid_message("g1")})
        fake_classifier = FakeClassifier(
            default=make_classification(category=Category.DIGEST, urgency=UrgencyLevel.LOW)
        )
        monkeypatch.setattr("app.gmail.client.GmailClient.from_token_file", lambda *a, **kw: fake_gmail_client)
        monkeypatch.setattr("app.ai.classifier.get_classifier", lambda settings: fake_classifier)

        at = run_app()
        at.button[0].click().run()
        at.button[0].click().run()

        assert fake_classifier.call_count == 1


class TestNoMutationControlsAnywhere:
    def test_no_delete_archive_reply_send_label_buttons_exist(self, db_path, token_path):
        connect(token_path)

        at = run_app()

        forbidden_words = ["delete", "archive", "reply", "send", "mark as read", "label", "trash"]
        button_labels = " ".join(b.label.lower() for b in at.button)
        assert not any(word in button_labels for word in forbidden_words)
