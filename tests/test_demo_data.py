"""Tests for app.ui.demo_data — the Inbox tab's one-click demo loader.

Uses a FakeClassifier and a temporary SQLite database, exactly like
tests/test_pipeline_dataset_run.py: no Gemini call, no network, no
production database touched.
"""

from pathlib import Path

from app.core.decisions import DecisionEngine
from app.core.pipeline import EmailProcessor
from app.evaluation.dataset import load_ground_truth
from app.preferences.manager import PreferenceManager
from app.storage.repositories import EmailRepository
from app.ui.demo_data import load_demo_emails, run_demo_batch
from tests.fakes import FakeClassifier, classification_for_dataset_record

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def _build_classifier(emails):
    ground_truth_by_id = {r.message_id: r for r in load_ground_truth()}
    classifications = {
        email.message_id: classification_for_dataset_record(
            email, {"expected_category": ground_truth_by_id[email.message_id].expected_category.value}
        )
        for email in emails
    }
    return FakeClassifier(classifications=classifications)


def test_load_demo_emails_respects_limit():
    emails = load_demo_emails(limit=5)

    assert len(emails) == 5
    assert len({e.message_id for e in emails}) == 5


def test_load_demo_emails_returns_first_n_dataset_records():
    all_records = load_ground_truth()
    emails = load_demo_emails(limit=3)

    assert [e.message_id for e in emails] == [r.message_id for r in all_records[:3]]


def test_run_demo_batch_stores_a_decision_for_every_demo_email(tmp_path):
    emails = load_demo_emails(limit=5)
    classifier = _build_classifier(emails)
    repository = EmailRepository(tmp_path / "demo.db")
    repository.initialize()
    processor = EmailProcessor(
        classifier=classifier,
        preferences=PreferenceManager.from_file(PREFERENCES_PATH),
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    results = run_demo_batch(processor, limit=5)

    assert len(results) == 5
    for email in emails:
        assert repository.get_decision(email.message_id) is not None


def test_run_demo_batch_is_safe_to_run_twice_without_reclassifying(tmp_path):
    emails = load_demo_emails(limit=5)
    classifier = _build_classifier(emails)
    repository = EmailRepository(tmp_path / "demo.db")
    repository.initialize()
    processor = EmailProcessor(
        classifier=classifier,
        preferences=PreferenceManager.from_file(PREFERENCES_PATH),
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    run_demo_batch(processor, limit=5)
    run_demo_batch(processor, limit=5)

    assert classifier.call_count == 5
