"""Verifies that DecisionEngine can consume every synthetic email in
data/eval_dataset.json alongside a deterministic (fake) ClassificationResult
— i.e. the same data structures M4's GeminiClassifier produces. No Gemini
calls are made.

This is an interface/compatibility check, not a correctness check: the
fake classifications below are derived mechanically from the dataset's own
`expected_category` purely so every code path in DecisionEngine gets
exercised across all 41 emails. The M5 spec is explicit that the dataset's
expected_category is classification-level ground truth, not decision-level
ground truth, so this file does not assert that DecisionEngine's output
must equal expected_category.
"""

from pathlib import Path

from app.core.decisions import DecisionEngine
from app.core.models import Category, ClassificationResult, RiskFlag, UrgencyLevel
from app.preferences.manager import PreferenceManager
from tests.test_eval_dataset import load_raw_dataset, to_email_message

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def _fake_classification(email, record) -> ClassificationResult:
    category = Category(record["expected_category"])
    risk_flags = [RiskFlag.OTHER] if category == Category.QUARANTINE else []
    urgency = UrgencyLevel.HIGH if category == Category.NOTIFY else UrgencyLevel.LOW
    return ClassificationResult(
        message_id=email.message_id,
        category=category,
        urgency=urgency,
        confidence=0.9,
        risk_flags=risk_flags,
        summary="Synthetic summary for decision-engine compatibility testing.",
        reasoning="Synthetic classification derived from dataset expected_category for compatibility testing only.",
    )


def test_decision_engine_accepts_every_synthetic_email():
    dataset = load_raw_dataset()
    engine = DecisionEngine()
    preferences = PreferenceManager.from_file(PREFERENCES_PATH)

    failures = []
    for record in dataset:
        try:
            email = to_email_message(record)
            classification = _fake_classification(email, record)
            decision = engine.decide(email, classification, preferences)
            assert decision.message_id == email.message_id
            assert decision.reasoning.strip() != ""
        except Exception as exc:  # pragma: no cover - failure path only
            failures.append((record["message_id"], str(exc)))

    assert failures == [], f"Records failed decision engine compatibility: {failures}"


def test_quarantine_ground_truth_records_always_decide_quarantine():
    """A stronger, still-safe assertion: every dataset record whose
    expected_category is QUARANTINE must still be decided as QUARANTINE
    here, because the fake classification for those records always carries
    a risk flag — this is really testing the security priority rule, using
    the dataset purely as a source of realistic inputs.
    """
    dataset = load_raw_dataset()
    engine = DecisionEngine()
    preferences = PreferenceManager.from_file(PREFERENCES_PATH)

    quarantine_records = [r for r in dataset if r["expected_category"] == "QUARANTINE"]
    assert quarantine_records, "dataset should contain QUARANTINE examples"

    for record in quarantine_records:
        email = to_email_message(record)
        classification = _fake_classification(email, record)
        decision = engine.decide(email, classification, preferences)
        assert decision.category == Category.QUARANTINE
