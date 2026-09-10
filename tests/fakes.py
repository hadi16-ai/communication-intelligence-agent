"""Shared test doubles. Not a test module itself (no test_ prefix) — it
holds fakes reused across several test files, in particular the pipeline
tests, which must never touch Gemini.
"""

from __future__ import annotations

from app.ai.classifier import BaseClassifier, ClassificationError
from app.core.models import Category, ClassificationResult, EmailMessage, RiskFlag, UrgencyLevel


class FakeClassifier(BaseClassifier):
    """A deterministic, network-free BaseClassifier for pipeline tests.

    Returns a pre-configured ClassificationResult per message_id (or a
    default template, with message_id substituted in) — it never talks to
    Gemini or any other network service. Tracks every call so tests can
    assert exactly how many times classification was actually requested,
    and for which message_ids — the whole point of testing the cache.
    """

    def __init__(
        self,
        classifications: dict[str, ClassificationResult] | None = None,
        default: ClassificationResult | None = None,
        fail_for: set[str] | None = None,
    ):
        self._classifications = classifications or {}
        self._default = default
        self._fail_for = fail_for or set()
        self.calls: list[str] = []

    def classify(self, email: EmailMessage) -> ClassificationResult:
        self.calls.append(email.message_id)

        if email.message_id in self._fail_for:
            raise ClassificationError(f"FakeClassifier configured to fail for {email.message_id!r}")

        if email.message_id in self._classifications:
            return self._classifications[email.message_id]

        if self._default is not None:
            return self._default.model_copy(update={"message_id": email.message_id})

        raise KeyError(f"FakeClassifier has no configured classification for {email.message_id!r}")

    @property
    def call_count(self) -> int:
        return len(self.calls)


def classification_for_dataset_record(email: EmailMessage, record: dict) -> ClassificationResult:
    """Derives a plausible ClassificationResult from one
    data/eval_dataset.json record's `expected_category` — used to build a
    FakeClassifier's canned responses for the full-dataset pipeline run.
    Mirrors the same approach used in the M5/M6 dataset compatibility
    tests: this is an interface-compatibility fixture, not a claim that
    the dataset's expected_category is decision-level ground truth.
    """
    category = Category(record["expected_category"])
    risk_flags = [RiskFlag.OTHER] if category == Category.QUARANTINE else []
    urgency = UrgencyLevel.HIGH if category == Category.NOTIFY else UrgencyLevel.LOW
    return ClassificationResult(
        message_id=email.message_id,
        category=category,
        urgency=urgency,
        confidence=0.9,
        risk_flags=risk_flags,
        summary="Synthetic summary for pipeline testing.",
        reasoning="Synthetic classification derived from dataset expected_category for pipeline testing only.",
    )
