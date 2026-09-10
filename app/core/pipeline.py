"""Orchestrates CACHE-LOOKUP -> CLASSIFY -> DECIDE -> STORE for one email,
wiring together the M2 preference manager, M4 classifier, M5 decision
engine, and M6 repository.

READ (where an EmailMessage comes from — synthetic fixtures now, Gmail in
M10) is intentionally outside this module's job: `EmailProcessor.process()`
takes an already-parsed `EmailMessage` and doesn't care about its source.

This module makes no Gemini/network call itself — it only calls whatever
`BaseClassifier` it was given, so tests can inject a fake and production
code can inject `GeminiClassifier` without either side changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ai.classifier import BaseClassifier
from app.core.decisions import DecisionEngine
from app.core.models import ClassificationResult, Decision, EmailMessage
from app.preferences.manager import PreferenceManager
from app.storage.repositories import EmailRepository


class ClassificationSource(str, Enum):
    """Where a processed email's classification came from. Exists purely
    for observability (e.g. a future dashboard showing "N served from
    cache") — it has no effect on the decision itself.
    """

    CACHE = "CACHE"
    CLASSIFIER = "CLASSIFIER"


@dataclass(frozen=True)
class ProcessingResult:
    """The outcome of processing one email: the final Decision plus
    whether its classification came from the cache or a fresh classifier
    call. Composes existing models rather than introducing a new one.
    """

    decision: Decision
    classification_source: ClassificationSource


class EmailProcessor:
    """Wires the classification cache (M6), classifier (M4), preference
    manager (M2), and decision engine (M5) together for one email at a
    time. Every dependency is injected so tests can substitute fakes
    without touching real Gemini, a real database, or real preferences.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        preferences: PreferenceManager,
        decision_engine: DecisionEngine,
        repository: EmailRepository,
    ):
        self._classifier = classifier
        self._preferences = preferences
        self._decision_engine = decision_engine
        self._repository = repository

    def process(self, email: EmailMessage) -> ProcessingResult:
        """Process one email end-to-end and return its final decision.

        Never invents a classification: if the classifier raises, or a
        stored record turns out to be inconsistent, this method lets that
        exception propagate rather than silently falling back to a
        fabricated result.
        """
        classification, source = self._get_or_create_classification(email)

        decision = self._decision_engine.decide(email, classification, self._preferences)
        self._repository.save_decision(decision)

        return ProcessingResult(decision=decision, classification_source=source)

    def _get_or_create_classification(
        self, email: EmailMessage
    ) -> tuple[ClassificationResult, ClassificationSource]:
        if self._repository.has_classification(email.message_id):
            cached = self._repository.get_classification(email.message_id)
            return cached, ClassificationSource.CACHE

        classification = self._classifier.classify(email)
        self._repository.save_classification(email, classification)
        return classification, ClassificationSource.CLASSIFIER


def run_batch(processor: EmailProcessor, emails: list[EmailMessage]) -> list[ProcessingResult]:
    """Processes a list of emails through `processor` in order, returning
    one `ProcessingResult` per email. A thin convenience wrapper —
    `process()` already does the real work for a single email; this just
    saves callers (the dataset runner, tests) a loop.
    """
    return [processor.process(email) for email in emails]
