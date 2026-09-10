"""Runs the full data/eval_dataset.json (41 synthetic emails) through
EmailProcessor end-to-end: cache lookup -> fake classifier when needed ->
preferences -> decision engine -> SQLite storage.

Uses a FakeClassifier (tests/fakes.py) and a temporary SQLite database —
no Gemini call, no network, and no production database are ever touched.
This is the automated equivalent of scripts/run_synthetic_pipeline.py,
which does the same thing against the real GeminiClassifier manually.
"""

from pathlib import Path

from app.core.decisions import DecisionEngine
from app.core.models import Category
from app.core.pipeline import ClassificationSource, EmailProcessor, run_batch
from app.preferences.manager import PreferenceManager
from app.storage.database import get_connection
from app.storage.repositories import EmailRepository
from tests.fakes import FakeClassifier, classification_for_dataset_record
from tests.test_eval_dataset import load_raw_dataset, to_email_message

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def _build_classifier_and_emails():
    dataset = load_raw_dataset()
    emails = [to_email_message(record) for record in dataset]
    classifications = {
        email.message_id: classification_for_dataset_record(email, record)
        for email, record in zip(emails, dataset)
    }
    return FakeClassifier(classifications=classifications), emails, dataset


def test_full_dataset_first_run_processes_all_41_successfully(tmp_path):
    classifier, emails, dataset = _build_classifier_and_emails()
    repository = EmailRepository(tmp_path / "full_run.db")
    repository.initialize()
    processor = EmailProcessor(
        classifier=classifier,
        preferences=PreferenceManager.from_file(PREFERENCES_PATH),
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    results = run_batch(processor, emails)

    assert len(results) == len(dataset) == 41
    assert classifier.call_count == 41  # every email was a cache miss first time
    assert all(result.classification_source == ClassificationSource.CLASSIFIER for result in results)

    for email in emails:
        assert repository.has_classification(email.message_id)
        assert repository.get_decision(email.message_id) is not None

    with get_connection(repository._db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM email_records").fetchone()["c"]
        message_ids = [row["message_id"] for row in conn.execute("SELECT message_id FROM email_records")]
    assert count == 41
    assert len(set(message_ids)) == 41  # no duplicates


def test_full_dataset_second_run_uses_cache_for_all_41(tmp_path):
    classifier, emails, dataset = _build_classifier_and_emails()
    repository = EmailRepository(tmp_path / "full_run_cached.db")
    repository.initialize()
    processor = EmailProcessor(
        classifier=classifier,
        preferences=PreferenceManager.from_file(PREFERENCES_PATH),
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    first_results = run_batch(processor, emails)
    assert classifier.call_count == 41

    second_results = run_batch(processor, emails)

    assert classifier.call_count == 41  # unchanged: no new classification calls
    assert all(result.classification_source == ClassificationSource.CACHE for result in second_results)

    with get_connection(repository._db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM email_records").fetchone()["c"]
        message_ids = [row["message_id"] for row in conn.execute("SELECT message_id FROM email_records")]
    assert count == 41
    assert len(set(message_ids)) == 41

    for first, second in zip(first_results, second_results):
        assert first.decision == second.decision


def test_quarantine_records_remain_quarantine_across_the_full_run(tmp_path):
    classifier, emails, dataset = _build_classifier_and_emails()
    repository = EmailRepository(tmp_path / "full_run_quarantine.db")
    repository.initialize()
    processor = EmailProcessor(
        classifier=classifier,
        preferences=PreferenceManager.from_file(PREFERENCES_PATH),
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    results = run_batch(processor, emails)

    expected_quarantine_ids = {r["message_id"] for r in dataset if r["expected_category"] == "QUARANTINE"}
    assert expected_quarantine_ids  # sanity: dataset really has some

    for email, result in zip(emails, results):
        if email.message_id in expected_quarantine_ids:
            assert result.decision.category == Category.QUARANTINE
