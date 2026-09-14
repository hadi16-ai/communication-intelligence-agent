"""Gemini implementation of BaseClassifier using structured output.

Reads the model name from Settings (GEMINI_MODEL) — never hard-coded here.
This module is the only place in the application that imports google-genai;
everything else depends on app.ai.classifier.BaseClassifier.
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.ai.classifier import BaseClassifier, ClassificationError
from app.ai.prompts import CLASSIFICATION_SYSTEM_INSTRUCTION, build_classification_prompt
from app.core.config import Settings
from app.core.models import Category, ClassificationResult, EmailMessage, RiskFlag, UrgencyLevel


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

    def classify(self, email: EmailMessage) -> ClassificationResult:
        try:
            response = self._generate_with_retry(email)
        except Exception as exc:
            raise ClassificationError(f"Gemini API call failed: {exc}") from exc

        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise ClassificationError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"Gemini response was not valid JSON: {exc}") from exc

        # The application, not the model, is the source of truth for
        # message_id — it isn't part of the schema we ask Gemini to fill in.
        payload["message_id"] = email.message_id

        try:
            return ClassificationResult.model_validate(payload)
        except Exception as exc:
            raise ClassificationError(
                f"Gemini response did not match the expected structured output: {exc}"
            ) from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
    )
    def _generate_with_retry(self, email: EmailMessage):
        """Retries transient failures only — `ServerError` (5xx) and HTTP
        429 `ClientError` (rate/quota limit) — with exponential backoff.
        Any other client error (bad request, auth failure, etc.) is not
        transient and is left to propagate immediately rather than retried.
        """
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
