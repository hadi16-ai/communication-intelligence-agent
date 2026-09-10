"""BaseClassifier abstract interface and classifier factory.

Keeps the rest of the application decoupled from any specific LLM
provider: everything outside app/ai should depend on BaseClassifier (and
ClassificationError), never on GeminiClassifier directly. Swapping Gemini
for Claude, OpenAI, or a local model later means adding a new module here
and updating get_classifier() — nothing else in the app should need to
change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.models import ClassificationResult, EmailMessage


class ClassificationError(RuntimeError):
    """Raised when a classifier cannot produce a valid ClassificationResult.

    Covers underlying API failures and responses that don't match the
    expected structured output. Never silently converted into a fake or
    default classification — callers must handle this explicitly.
    """


class BaseClassifier(ABC):
    """Understands an email and produces a ClassificationResult.

    Implementations must not make the final personalized routing decision
    (NOTIFY/DIGEST/MUTE/QUARANTINE for this specific user) — that is
    app/core/decisions.py's job. A classifier only reports what it
    understood about the email.
    """

    @abstractmethod
    def classify(self, email: EmailMessage) -> ClassificationResult:
        raise NotImplementedError


def get_classifier(settings: Settings | None = None) -> BaseClassifier:
    """Factory returning the configured classifier implementation.

    Accepts an explicit `settings` for testability; defaults to reading
    the real environment/.env otherwise. Imports GeminiClassifier lazily
    so that code depending only on BaseClassifier (e.g. the decision
    engine, most tests) never needs the google-genai package installed.
    """
    from app.ai.gemini import GeminiClassifier

    return GeminiClassifier(settings or Settings())
