"""Runs the full data/eval_dataset.json synthetic dataset through the
EmailProcessor pipeline (cache -> classify -> decide -> store), using the
REAL GeminiClassifier.

This is a manual, opt-in script — it is never invoked by the automated
test suite. The test suite exercises the exact same pipeline with a fake
classifier instead (see tests/test_pipeline_dataset_run.py) so it never
needs a real GEMINI_API_KEY or network access.

Requires a real GEMINI_API_KEY (see .env.example). Writes to the
configured database (Settings.database_path, default data/app.db).

Usage:
    python scripts/run_synthetic_pipeline.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.ai.classifier import get_classifier  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.decisions import DecisionEngine  # noqa: E402
from app.core.models import EmailMessage  # noqa: E402
from app.core.pipeline import EmailProcessor  # noqa: E402
from app.preferences.manager import PreferenceManager  # noqa: E402
from app.storage.repositories import EmailRepository  # noqa: E402

DATASET_PATH = REPO_ROOT / "data" / "eval_dataset.json"
PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def _to_email_message(record: dict) -> EmailMessage:
    return EmailMessage(
        message_id=record["message_id"],
        sender=record["sender"],
        subject=record["subject"],
        body=record["body"],
        received_at=datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")),
    )


def main() -> None:
    settings = Settings()
    records = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    emails = [_to_email_message(record) for record in records]

    preferences = PreferenceManager.from_file(PREFERENCES_PATH)
    repository = EmailRepository(settings.database_path)
    repository.initialize()
    processor = EmailProcessor(
        classifier=get_classifier(settings),
        preferences=preferences,
        decision_engine=DecisionEngine(),
        repository=repository,
    )

    category_counts: Counter = Counter()
    source_counts: Counter = Counter()
    failures = []

    for email in emails:
        try:
            result = processor.process(email)
        except Exception as exc:
            failures.append((email.message_id, str(exc)))
            continue
        category_counts[result.decision.category.value] += 1
        source_counts[result.classification_source.value] += 1

    print(f"Processed {len(emails)} emails ({len(failures)} failures)")
    print("Decisions by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:12s} {count}")
    print("Classification source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:12s} {count}")
    if failures:
        print("Failures:")
        for message_id, error in failures:
            print(f"  {message_id}: {error}")


if __name__ == "__main__":
    main()
