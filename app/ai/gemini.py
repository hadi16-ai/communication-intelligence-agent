"""Gemini implementation of BaseClassifier using structured output.

Reads the model name from Settings (GEMINI_MODEL) — never hard-coded here.
This module is the only place in the application that imports google-genai;
everything else depends on app.ai.classifier.BaseClassifier.
"""

from __future__ import annotations

import json
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.ai.classifier import BaseClassifier, ClassificationError
from app.ai.prompts import CLASSIFICATION_SYSTEM_INSTRUCTION, build_classification_prompt
from app.core.config import Settings
from app.core.models import CallMetrics, Category, ClassificationResult, EmailMessage, RiskFlag, UrgencyLevel
from app.utils.logging import get_logger

logger = get_logger(__name__)

# gemini-3.8-flash pricing (per Google's published Gemini API pricing for
# this model specifically — not copied from a different model's rate).
# This is the introductory rate in effect through 2026-12-31; standard
# pricing ($1.50 / $7.50 per million input/output tokens) takes effect
# 2027-01-01 and will need updating here when it does.
INPUT_COST_PER_MILLION_TOKENS_USD = 0.75
OUTPUT_COST_PER_MILLION_TOKENS_USD = 3.75


def _build_response_schema() -> dict:
    """The JSON schema Gemini must fill in.

    Deliberately hand-written rather than derived from
    `ClassificationResult.model_json_schema()`: pydantic emits `$defs`/`$ref`
    for the enum fields, and Gemini's structured-output support only
    handles a restricted, non-referencing subset of OpenAPI 3.0. A flat
    schema avoids that incompatibility entirely. `message_id` is
    intentionally excluded — the model has no reliable way to know it, so
    the application supplies it itself after parsing the response.
    Enum choices are still derived from the real enums so this can't drift
    out of sync with app.core.models.
    """
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": [c.value for c in Category]},
            "urgency": {"type": "string", "enum": [u.value for u in UrgencyLevel]},
            "confidence": {"type": "number"},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": [r.value for r in RiskFlag]},
            },
            "summary": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["category", "urgency", "confidence", "summary", "reasoning"],
    }


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures worth a retry: any `ServerError` (5xx — model
    overloaded/unavailable), plus specifically HTTP 429 `ClientError`
    (RESOURCE_EXHAUSTED — rate/quota limit). Other `ClientError`s (bad
    request, auth failure) are not transient and must propagate
    immediately rather than waste retries on something retrying can't fix.
    """
    if isinstance(exc, ServerError):
        return True
    return isinstance(exc, ClientError) and getattr(exc, "code", None) == 429


def _extract_token_usage(response) -> tuple[int | None, int | None]:
    """Real per-call token usage from `response.usage_metadata`.

    Field names here were verified against a live API response
    (`GenerateContentResponseUsageMetadata`) rather than assumed:
    `prompt_token_count` (input) and `candidates_token_count` (visible
    output) are the obvious ones, but a real response also carries a
    `thoughts_token_count` for this model's internal reasoning — tokens
    that never appear in the visible response text but ARE billed as
    output tokens. Omitting them would understate cost substantially (in
    one verified sample call: 4 visible output tokens vs. 81 thinking
    tokens). Returns (None, None) only if the SDK reports no usage data
    at all — never fabricates 0 in that case.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    input_tokens = usage.prompt_token_count or 0
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return input_tokens, output_tokens


def _compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION_TOKENS_USD
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION_TOKENS_USD
    )


class GeminiClassifier(BaseClassifier):
    """Classifies emails using Google's Gemini API.

    Never makes the final personalized routing decision, and never falls
    back to a fabricated classification on failure — API and validation
    errors are always raised as `ClassificationError`.
    """

    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        if settings.gemini_api_key is None:
            raise ClassificationError(
                "GEMINI_API_KEY is not configured; cannot create GeminiClassifier."
            )
        self._model = settings.gemini_model
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._last_metrics: CallMetrics | None = None

    @property
    def last_call_metrics(self) -> CallMetrics | None:
        return self._last_metrics

    def classify(self, email: EmailMessage) -> ClassificationResult:
        start = time.perf_counter()
        try:
            response, retry_count = self._generate_with_retry(email)
        except Exception as exc:
            self._fail(email, start, retry_count=getattr(exc, "retry_count", 0), error=exc)
            raise ClassificationError(f"Gemini API call failed: {exc}") from exc

        input_tokens, output_tokens = _extract_token_usage(response)
        cost_usd = (
            _compute_cost_usd(input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        token_kwargs = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}

        raw_text = getattr(response, "text", None)
        if not raw_text:
            self._fail(email, start, retry_count=retry_count, error=ClassificationError("empty response"), **token_kwargs)
            raise ClassificationError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self._fail(email, start, retry_count=retry_count, error=exc, **token_kwargs)
            raise ClassificationError(f"Gemini response was not valid JSON: {exc}") from exc

        # The application, not the model, is the source of truth for
        # message_id — it isn't part of the schema we ask Gemini to fill in.
        payload["message_id"] = email.message_id

        try:
            result = ClassificationResult.model_validate(payload)
        except Exception as exc:
            self._fail(email, start, retry_count=retry_count, error=exc, **token_kwargs)
            raise ClassificationError(
                f"Gemini response did not match the expected structured output: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = CallMetrics(
            latency_ms=latency_ms,
            retry_count=retry_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        logger.info(
            "classification succeeded message_id=%s category=%s retry_count=%d latency_ms=%.1f "
            "input_tokens=%s output_tokens=%s cost_usd=%s",
            email.message_id, result.category.value, retry_count, latency_ms,
            input_tokens, output_tokens, f"{cost_usd:.6f}" if cost_usd is not None else "n/a",
        )
        return result

    def _fail(
        self,
        email: EmailMessage,
        start: float,
        *,
        retry_count: int,
        error: BaseException,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Records metrics and logs an ERROR for a call that ultimately
        failed. Logs only message_id, error type/message, and numeric
        metrics — the error message here always originates from the
        Gemini API/SDK or JSON/schema validation, never from email
        content, so it is safe to log in full.
        """
        latency_ms = (time.perf_counter() - start) * 1000
        error_type = type(error).__name__
        self._last_metrics = CallMetrics(
            latency_ms=latency_ms,
            retry_count=retry_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            error_type=error_type,
        )
        logger.error(
            "classification failed message_id=%s error_type=%s retry_count=%d latency_ms=%.1f message=%s",
            email.message_id, error_type, retry_count, latency_ms, str(error),
        )

    def _generate_with_retry(self, email: EmailMessage) -> tuple[object, int]:
        """Retries transient failures only — `ServerError` (5xx) and HTTP
        429 `ClientError` (rate/quota limit) — with exponential backoff.
        Any other client error (bad request, auth failure, etc.) is not
        transient and is left to propagate immediately rather than retried.

        Uses a fresh `Retrying` instance per call (rather than the
        `@retry` decorator) so `.statistics['attempt_number']` — used to
        derive `retry_count` — is call-local, not shared/mutated state on
        the decorated method across concurrent or sequential calls.
        """

        def _before_sleep(retry_state) -> None:
            exc = retry_state.outcome.exception()
            logger.warning(
                "classification retry message_id=%s attempt=%d error_type=%s",
                email.message_id, retry_state.attempt_number, type(exc).__name__,
            )

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception(_is_retryable),
            before_sleep=_before_sleep,
        )
        try:
            response = retrying(self._call_gemini, email)
        except Exception as exc:
            exc.retry_count = retrying.statistics.get("attempt_number", 1) - 1
            raise
        retry_count = retrying.statistics.get("attempt_number", 1) - 1
        return response, retry_count

    def _call_gemini(self, email: EmailMessage):
        return self._client.models.generate_content(
            model=self._model,
            contents=build_classification_prompt(email),
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFICATION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=_build_response_schema(),
                temperature=0.0,
            ),
        )
