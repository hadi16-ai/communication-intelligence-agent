"""Verifies that every synthetic email in data/eval_dataset.json can be
stored and retrieved through EmailRepository. No Gemini calls — a
deterministic fake classification is generated per record purely to
exercise the storage layer, mirroring the same approach used for the M5
decision-engine dataset compatibility test.
"""

from app.core.models import Category, ClassificationResult, RiskFlag, UrgencyLevel
from app.storage.repositories import EmailRepository
from tests.test_eval_dataset import load_raw_dataset, to_email_message


def _fake_classification(email, record) -> ClassificationResult:
    category = Category(record["expected_category"])
    risk_flags = [RiskFlag.OTHER] if category == Category.QUARANTINE else []
    urgency = UrgencyLevel.HIGH if category == Category.NOTIFY else UrgencyLevel.LOW
    return ClassificationResult(
        message_id=email.message_id,
        category=category,
        urgency=urgency,
        confidence=0.9,
        risk_flags=risk_flags,
        summary="Synthetic summary for storage compatibility testing.",
        reasoning="Synthetic classification derived from dataset expected_category for compatibility testing only.",
    )


def test_every_synthetic_email_can_be_stored_and_retrieved(tmp_path):
    dataset = load_raw_dataset()
    repo = EmailRepository(tmp_path / "dataset_compat.db")
    repo.initialize()

    failures = []
    for record in dataset:
        try:
            email = to_email_message(record)
            classification = _fake_classification(email, record)

            repo.save_classification(email, classification)
            stored = repo.get_classification(email.message_id)

            assert stored == classification
            assert repo.has_classification(email.message_id) is True
        except Exception as exc:  # pragma: no cover - failure path only
            failures.append((record["message_id"], str(exc)))

    assert failures == [], f"Records failed storage compatibility: {failures}"


def test_all_41_records_produce_distinct_rows_with_no_duplicates(tmp_path):
    dataset = load_raw_dataset()
    repo = EmailRepository(tmp_path / "dataset_compat_dupes.db")
    repo.initialize()

    for record in dataset:
        email = to_email_message(record)
        classification = _fake_classification(email, record)
        repo.save_classification(email, classification)

    from app.storage.database import get_connection

    with get_connection(repo._db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM email_records").fetchone()["c"]

    assert count == len(dataset)
