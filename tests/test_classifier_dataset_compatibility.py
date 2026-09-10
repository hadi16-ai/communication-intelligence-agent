"""Verifies that every synthetic email in data/eval_dataset.json can be
converted to an EmailMessage and accepted by the BaseClassifier interface.

Uses a fake, in-memory classifier — this is a compatibility check for the
classifier *interface*, not a correctness check of real Gemini output, and
makes no network calls.
"""

from app.ai.classifier import BaseClassifier
from app.core.models import Category, ClassificationResult, EmailMessage, UrgencyLevel
from tests.test_eval_dataset import load_raw_dataset, to_email_message


class FakeClassifier(BaseClassifier):
    def classify(self, email: EmailMessage) -> ClassificationResult:
        return ClassificationResult(
            message_id=email.message_id,
            category=Category.DIGEST,
            urgency=UrgencyLevel.LOW,
            confidence=0.5,
            risk_flags=[],
            summary="stub",
            reasoning="stub",
        )


def test_every_synthetic_email_is_accepted_by_the_classifier_interface():
    dataset = load_raw_dataset()
    classifier: BaseClassifier = FakeClassifier()

    failures = []
    for record in dataset:
        try:
            email = to_email_message(record)
            result = classifier.classify(email)
            assert result.message_id == email.message_id
        except Exception as exc:  # pragma: no cover - failure path only
            failures.append((record["message_id"], str(exc)))

    assert failures == [], f"Records failed classifier compatibility: {failures}"
