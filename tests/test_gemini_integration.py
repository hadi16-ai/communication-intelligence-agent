"""Optional, explicit opt-in integration test that makes a REAL call to the
Gemini API.

This test is skipped by default and is NOT part of the normal test suite
run. It only runs when both of the following are true:

  1. RUN_GEMINI_LIVE_TESTS=1 is set (an explicit, deliberate opt-in — just
     having a real GEMINI_API_KEY configured for normal app use is not
     enough to trigger a live call during `pytest`).
  2. A real GEMINI_API_KEY is present in the environment/.env.

Run it explicitly with:

    RUN_GEMINI_LIVE_TESTS=1 pytest tests/test_gemini_integration.py -v

This costs a small amount of real Gemini API quota.
"""

import os
from datetime import datetime, timezone

import pytest

from app.ai.gemini import GeminiClassifier
from app.core.config import Settings
from app.core.models import ClassificationResult, EmailMessage

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_LIVE_TESTS") != "1",
    reason="Live Gemini integration test skipped by default. "
    "Set RUN_GEMINI_LIVE_TESTS=1 (and a real GEMINI_API_KEY) to run it.",
)


def test_real_gemini_classifies_an_obvious_phishing_email():
    settings = Settings()
    if settings.gemini_api_key is None:
        pytest.skip("No real GEMINI_API_KEY configured; skipping live call.")

    classifier = GeminiClassifier(settings)
    email = EmailMessage(
        message_id="live-test-1",
        sender="security@totally-not-a-bank-verify.invalid",
        subject="Your account will be suspended - verify now",
        body=(
            "We were unable to verify your account. Click here immediately "
            "to confirm your password or your account will be suspended: "
            "http://verify-now.invalid/login"
        ),
        received_at=datetime.now(timezone.utc),
    )

    result = classifier.classify(email)

    assert isinstance(result, ClassificationResult)
    assert result.message_id == "live-test-1"
