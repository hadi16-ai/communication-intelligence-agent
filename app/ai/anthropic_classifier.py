"""Claude implementation of BaseClassifier using forced tool-use for
structured output.

Reads the model name from Settings (ANTHROPIC_MODEL) — never hard-coded
inline. This module is the only place in the application that imports the
`anthropic` SDK; everything else depends on app.ai.classifier.BaseClassifier.

Reuses, rather than duplicates, the same provider-agnostic pieces the
Gemini classifier already uses: the injection-hardened prompt text in
app/ai/prompts.py, and `ClassificationResult.model_validate()` for
validating the parsed response — there is exactly one place either
classifier's output gets validated against the domain schema.
"""

from __future__ import annotations

import time

import anthropic

from app.ai.classifier import BaseClassifier, ClassificationError
from app.ai.prompts import CLASSIFICATION_SYSTEM_INSTRUCTION, build_classification_prompt
from app.core.config import Settings
from app.core.models import CallMetrics, Category, ClassificationResult, EmailMessage, RiskFlag, UrgencyLevel
from app.utils.logging import get_logger
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

logger = get_logger(__name__)

# Claude Haiku 4.5 pricing (per Anthropic's published API pricing for this
# model specifically — not copied from a different model's rate).
INPUT_COST_PER_MILLION_TOKENS_USD = 1.00
OUTPUT_COST_PER_MILLION_TOKENS_USD = 5.00

CLASSIFICATION_TOOL_NAME = "submit_email_classification"


def _build_classification_tool() -> dict:
    """The forced tool-use schema Claude must fill in — Anthropic's
    documented reliable pattern for structured output (rather than
    prompting-and-hoping the model returns parseable JSON). Mirrors
    app/ai/gemini.py's `_build_response_schema()` field-for-field so both
    providers are asked for exactly the same shape; `message_id` is
    excluded for the same reason it is there — the model has no reliable
    way to know it, so the application supplies it itself afterward.
    """
    return {
        "name": CLASSIFICATION_TOOL_NAME,
        "description": "Submit the classification for the analyzed email.",
        "input_schema": {
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
        },
    }


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures worth a retry: `RateLimitError` (429) and any
    `APIStatusError` with a 5xx status (server-side/overloaded). Other
    `APIStatusError`s (400 bad request, 401/403 auth/permission, 404
    model-not-found) are not transient and must propagate immediately —
    identical policy to app/ai/gemini.py's `_is_retryable`, adapted to the
    anthropic SDK's exception hierarchy.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def _extract_token_usage(response) -> tuple[int | None, int | None]:
    """Real per-call token usage from `response.usage` — field names
    (`input_tokens`, `output_tokens`) verified directly against the
    installed anthropic SDK's `Usage` type rather than assumed. Unlike
    Gemini, Claude's `output_tokens` already represents the full billed
    output (there is no separate "thinking tokens" field to add in for a
    non-extended-thinking call such as this one). Returns (None, None)
    only if the SDK reports no usage data at all — never fabricates 0.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return usage.input_tokens, usage.output_tokens


def _compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION_TOKENS_USD
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION_TOKENS_USD
    )


class AnthropicClassifier(BaseClassifier):
    """Classifies emails using Anthropic's Claude API.

    Never makes the final personalized routing decision, and never falls
    back to a fabricated classification on failure — API and validation
    errors are always raised as `ClassificationError`. Structural mirror
    of GeminiClassifier (app/ai/gemini.py): same retry policy shape, same
    CallMetrics population, same logging discipline (IDs/categories/
    numeric metrics only, never email content).
    """

    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        if settings.anthropic_api_key is None:
            raise ClassificationError(
                "ANTHROPIC_API_KEY is not configured; cannot create AnthropicClassifier."
            )
        self._model = settings.anthropic_model
        # max_retries=0: this classifier owns retry policy itself (via the
        # same Retrying-per-call pattern as GeminiClassifier) so behavior
        # and retry_count are identical in shape across both providers,
        # rather than layering the SDK's own default retry on top.
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), max_retries=0
        )
        self._last_metrics: CallMetrics | None = None

    @property
    def last_call_metrics(self) -> CallMetrics | None:
        return self._last_metrics

    def classify(self, email: EmailMessage) -> ClassificationResult:
        start = time.perf_counter()
        try:
            response, retry_count = self._create_with_retry(email)
        except Exception as exc:
            self._fail(email, start, retry_count=getattr(exc, "retry_count", 0), error=exc)
            raise ClassificationError(f"Claude API call failed: {exc}") from exc

        input_tokens, output_tokens = _extract_token_usage(response)
        cost_usd = (
            _compute_cost_usd(input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        token_kwargs = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}

        tool_use_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"), None
        )
        if tool_use_block is None:
            self._fail(
                email, start, retry_count=retry_count,
                error=ClassificationError("no tool_use block in response despite forced tool_choice"),
                **token_kwargs,
            )
            raise ClassificationError("Claude did not return the forced tool call.")

        payload = dict(tool_use_block.input)
        # The application, not the model, is the source of truth for
        # message_id — it isn't part of the tool schema we force Claude to fill in.
        payload["message_id"] = email.message_id

        try:
            result = ClassificationResult.model_validate(payload)
        except Exception as exc:
            self._fail(email, start, retry_count=retry_count, error=exc, **token_kwargs)
            raise ClassificationError(
                f"Claude response did not match the expected structured output: {exc}"
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
        Claude API/SDK or JSON/schema validation, never from email
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

    def _create_with_retry(self, email: EmailMessage) -> tuple[object, int]:
        """Retries transient failures only — `RateLimitError` (429) and
        5xx `APIStatusError` — with exponential backoff. Any other error
        (bad request, auth failure, model-not-found, etc.) is not
        transient and is left to propagate immediately rather than
        retried. Same fresh-`Retrying`-instance-per-call approach as
        GeminiClassifier, for the same reason: `retry_count` must be
        call-local, not shared/mutated state.
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
            response = retrying(self._call_claude, email)
        except Exception as exc:
            exc.retry_count = retrying.statistics.get("attempt_number", 1) - 1
            raise
        retry_count = retrying.statistics.get("attempt_number", 1) - 1
        return response, retry_count

    def _call_claude(self, email: EmailMessage):
        # No `temperature` param: this SDK version's Messages.create()
        # doesn't accept one at all (verified directly against the
        # installed SDK's signature — it isn't a removed default, there's
        # simply no sampling-temperature knob in this API shape). Forced
        # tool_choice against a small fixed schema already constrains
        # output significantly without it.
        return self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=CLASSIFICATION_SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": build_classification_prompt(email)}],
            tools=[_build_classification_tool()],
            tool_choice={"type": "tool", "name": CLASSIFICATION_TOOL_NAME},
        )
