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
from app.core.models import CallMetrics, ClassificationResult, EmailMessage


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

    @property
    def last_call_metrics(self) -> CallMetrics | None:
        """Observability metrics for the most recent `classify()` call, if
        this implementation tracks them — `None` by default (e.g. test
        fakes), which callers must treat as "no data," never as zero
        cost/latency. Set even when `classify()` raises, so a caller can
        inspect latency/retry_count/error_type after catching
        `ClassificationError`.
        """
        return None


def get_classifier(settings: Settings | None = None) -> BaseClassifier:
    """Factory returning the configured classifier implementation,
    selected by `Settings.ai_provider` ("gemini" or "anthropic" —
    defaults to "anthropic").

    Accepts an explicit `settings` for testability; defaults to reading
    the real environment/.env otherwise. Imports each provider's module
    lazily, and only the one actually selected, so code depending only on
    BaseClassifier (e.g. the decision engine, most tests) never needs
    either google-genai or anthropic installed, and selecting one
    provider never requires the other provider's SDK to be present.
    """
    settings = settings or Settings()
    provider = settings.ai_provider.lower()

    if provider == "gemini":
        from app.ai.gemini import GeminiClassifier

        return GeminiClassifier(settings)

    if provider == "anthropic":
        from app.ai.anthropic_classifier import AnthropicClassifier

        return AnthropicClassifier(settings)

    raise ClassificationError(
        f"Unknown AI_PROVIDER {provider!r}; expected 'gemini' or 'anthropic'."
    )
