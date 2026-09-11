"""Tests for app.ui.gmail_view: connection-status check and
fetch-and-process orchestration. No real Gmail API calls, no Gemini —
uses a fake GmailClient and a real EmailProcessor wired to a fake
classifier and a temporary SQLite database.
"""

import pytest

from app.core.decisions import DecisionEngine
from app.core.models import Category, UrgencyLevel
from app.core.pipeline import ClassificationSource, EmailProcessor
from app.gmail.client import FetchResult
from app.preferences.manager import PreferenceManager, Preferences
from app.storage.repositories import EmailRepository
from app.ui.gmail_view import format_sync_summary, gmail_is_connected, sync_gmail
from tests.fakes import FakeClassifier
from tests.test_gmail_client import FakeGmailClient, valid_message
from tests.test_models import make_classification


@pytest.fixture
def repo(tmp_path) -> EmailRepository:
    repository = EmailRepository(tmp_path / "gmail_view_test.db")
    repository.initialize()
    return repository


@pytest.fixture
def preferences() -> PreferenceManager:
    return PreferenceManager(Preferences())


class TestGmailIsConnected:
    def test_false_when_token_file_missing(self, tmp_path):
        assert gmail_is_connected(tmp_path / "does_not_exist.json") is False

    def test_true_when_token_file_exists(self, tmp_path):
        token_path = tmp_path / "token.json"
        token_path.write_text("{}", encoding="utf-8")

        assert gmail_is_connected(token_path) is True


class TestSyncGmail:
    def test_fetches_and_processes_through_the_real_pipeline(self, repo, preferences):
        gmail_client = FakeGmailClient(
            ids=["g1", "g2"],
            messages={"g1": valid_message("g1", "Hello"), "g2": valid_message("g2", "World")},
        )
        classifier = FakeClassifier(
            default=make_classification(category=Category.DIGEST, urgency=UrgencyLevel.LOW)
        )
        processor = EmailProcessor(
            classifier=classifier, preferences=preferences, decision_engine=DecisionEngine(), repository=repo
        )

        result = sync_gmail(gmail_client, processor, max_results=10)

        assert result.fetched_count == 2
        assert result.failed_count == 0
        assert len(result.processing) == 2
        assert repo.has_classification("g1")
        assert repo.has_classification("g2")

    def test_uses_the_existing_cache_not_a_parallel_one(self, repo, preferences):
        """Processing the same Gmail message_id twice must not call the
        classifier twice — proving Gmail sync reuses EmailProcessor's
        existing M6 cache rather than any new caching logic.
        """
        gmail_client = FakeGmailClient(ids=["g1"], messages={"g1": valid_message("g1")})
        classifier = FakeClassifier(
            default=make_classification(category=Category.DIGEST, urgency=UrgencyLevel.LOW)
        )
        processor = EmailProcessor(
            classifier=classifier, preferences=preferences, decision_engine=DecisionEngine(), repository=repo
        )

        sync_gmail(gmail_client, processor, max_results=10)
        result_2 = sync_gmail(gmail_client, processor, max_results=10)

        assert classifier.call_count == 1
        assert result_2.cache_hit_count == 1
        assert result_2.newly_classified_count == 0

    def test_gmail_fetch_failures_are_reported_not_silently_dropped(self, repo, preferences):
        from app.gmail.client import GmailAPIError

        gmail_client = FakeGmailClient(
            ids=["good", "bad"],
            messages={"good": valid_message("good")},
            get_errors={"bad": GmailAPIError("simulated failure")},
        )
        classifier = FakeClassifier(
            default=make_classification(category=Category.DIGEST, urgency=UrgencyLevel.LOW)
        )
        processor = EmailProcessor(
            classifier=classifier, preferences=preferences, decision_engine=DecisionEngine(), repository=repo
        )

        result = sync_gmail(gmail_client, processor, max_results=10)

        assert result.fetched_count == 1
        assert result.failed_count == 1
        assert result.fetch.failed_message_ids[0][0] == "bad"

    def test_no_raw_body_ends_up_in_sqlite_after_gmail_sync(self, repo, preferences):
        gmail_client = FakeGmailClient(ids=["g1"], messages={"g1": valid_message("g1", "Secret body content")})
        classifier = FakeClassifier(
            default=make_classification(category=Category.DIGEST, urgency=UrgencyLevel.LOW)
        )
        processor = EmailProcessor(
            classifier=classifier, preferences=preferences, decision_engine=DecisionEngine(), repository=repo
        )

        sync_gmail(gmail_client, processor, max_results=10)

        from app.storage.database import get_connection

        with get_connection(repo._db_path) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(email_records)")}
            row = conn.execute("SELECT * FROM email_records WHERE message_id = 'g1'").fetchone()

        assert "body" not in columns
        assert "Secret body content" not in str(dict(row))


class TestFormatSyncSummary:
    def test_summarizes_counts(self):
        from app.ui.gmail_view import GmailSyncResult
        from app.core.models import Decision, DecisionSource

        classification = make_classification(message_id="g1", category=Category.DIGEST)
        decision = Decision(
            message_id="g1",
            category=Category.DIGEST,
            reasoning="test",
            source=DecisionSource.AI_CLASSIFICATION,
            classification=classification,
        )
        from app.core.pipeline import ProcessingResult

        result = GmailSyncResult(
            fetch=FetchResult(emails=[], failed_message_ids=[("bad-1", "boom")]),
            processing=[ProcessingResult(decision=decision, classification_source=ClassificationSource.CACHE)],
        )

        summary = format_sync_summary(result)

        assert "1 message(s) could not be processed" in summary
