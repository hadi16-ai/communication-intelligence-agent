"""Tests for the BaseClassifier abstraction and the classifier factory.

No network calls anywhere in this file, and no dependency on a real
GEMINI_API_KEY — every Settings() is constructed with `_env_file=None` and
explicit values so these tests are fully isolated from the developer's
real environment.
"""

from datetime import datetime, timezone

import pytest

from app.ai.classifier import BaseClassifier, ClassificationError, get_classifier
from app.core.config import Settings
from app.core.models import Category, ClassificationResult, EmailMessage, UrgencyLevel


def make_email(**overrides) -> EmailMessage:
    defaults = dict(
        message_id="msg-1",
        sender="someone@example.com",
        subject="Hello",
        body="Just checking in.",
        received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class FakeClassifier(BaseClassifier):
    """A minimal, deterministic BaseClassifier used to test the interface
    itself, independent of Gemini.
    """

    def classify(self, email: EmailMessage) -> ClassificationResult:
        return ClassificationResult(
            message_id=email.message_id,
            category=Category.DIGEST,
            urgency=UrgencyLevel.LOW,
            confidence=0.5,
            risk_flags=[],
            summary="stub summary",
            reasoning="stub reasoning",
        )


def test_base_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseClassifier()


def test_subclass_implementing_classify_can_be_used_polymorphically():
    classifier: BaseClassifier = FakeClassifier()

    result = classifier.classify(make_email())

    assert isinstance(result, ClassificationResult)
    assert result.category == Category.DIGEST


def test_get_classifier_without_api_key_raises_classification_error():
    settings = Settings(_env_file=None, gemini_api_key=None)

    with pytest.raises(ClassificationError):
        get_classifier(settings)


def test_get_classifier_with_api_key_returns_a_base_classifier():
    settings = Settings(_env_file=None, gemini_api_key="fake-test-key-not-real")

    classifier = get_classifier(settings)

    assert isinstance(classifier, BaseClassifier)
