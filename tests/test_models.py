"""Tests for app.core.models: EmailMessage, ClassificationResult, Decision,
and the fixed-choice enums (Category, UrgencyLevel, RiskFlag, DecisionSource).
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.models import (
    Category,
    ClassificationResult,
    Decision,
    DecisionSource,
    EmailMessage,
    RiskFlag,
    UrgencyLevel,
)


def make_email(**overrides) -> EmailMessage:
    defaults = dict(
        message_id="msg-1",
        sender="alice@example.com",
        subject="Interview scheduling",
        body="Are you available Thursday at 2pm?",
        received_at=datetime(2026, 1, 5, 12, 30, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


def make_classification(**overrides) -> ClassificationResult:
    defaults = dict(
        message_id="msg-1",
        category=Category.NOTIFY,
        urgency=UrgencyLevel.HIGH,
        confidence=0.92,
        risk_flags=[],
        summary="Interviewer proposing a call time.",
        reasoning="Sender is an interviewer discussing scheduling.",
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


def make_decision(**overrides) -> Decision:
    classification = overrides.pop("classification", None) or make_classification()
    defaults = dict(
        message_id=classification.message_id,
        category=classification.category,
        reasoning="Matches AI classification; no preference override applied.",
        source=DecisionSource.AI_CLASSIFICATION,
        classification=classification,
    )
    defaults.update(overrides)
    return Decision(**defaults)


# --- EmailMessage -----------------------------------------------------


class TestEmailMessage:
    def test_valid_construction(self):
        email = make_email()
        assert email.message_id == "msg-1"
        assert email.sender == "alice@example.com"
        assert email.thread_id is None

    def test_subject_and_body_default_to_empty_string(self):
        email = EmailMessage(
            message_id="msg-2",
            sender="bob@example.com",
            received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert email.subject == ""
        assert email.body == ""

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            EmailMessage(sender="bob@example.com", received_at=datetime.now(timezone.utc))

    def test_empty_message_id_rejected(self):
        with pytest.raises(ValidationError):
            make_email(message_id="")

    def test_empty_sender_rejected(self):
        with pytest.raises(ValidationError):
            make_email(sender="")

    def test_serialization_round_trip(self):
        email = make_email()
        restored = EmailMessage.model_validate_json(email.model_dump_json())
        assert restored == email


# --- ClassificationResult ----------------------------------------------


class TestClassificationResult:
    def test_valid_construction(self):
        result = make_classification()
        assert result.category == Category.NOTIFY
        assert result.urgency == UrgencyLevel.HIGH
        assert result.risk_flags == []

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationResult(
                message_id="msg-1",
                urgency=UrgencyLevel.LOW,
                confidence=0.5,
                summary="x",
                reasoning="x",
            )

    @pytest.mark.parametrize("bad_category", ["SPAM", "URGENT", "notify", ""])
    def test_invalid_category_rejected(self, bad_category):
        with pytest.raises(ValidationError):
            make_classification(category=bad_category)

    @pytest.mark.parametrize("bad_urgency", ["CRITICAL", "low", ""])
    def test_invalid_urgency_rejected(self, bad_urgency):
        with pytest.raises(ValidationError):
            make_classification(urgency=bad_urgency)

    def test_invalid_risk_flag_rejected(self):
        with pytest.raises(ValidationError):
            make_classification(risk_flags=["NOT_A_REAL_FLAG"])

    def test_valid_risk_flags_accepted(self):
        result = make_classification(
            category=Category.QUARANTINE,
            risk_flags=[RiskFlag.PHISHING, RiskFlag.SUSPICIOUS_LINK],
        )
        assert result.risk_flags == [RiskFlag.PHISHING, RiskFlag.SUSPICIOUS_LINK]

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -1.0])
    def test_confidence_out_of_range_rejected(self, confidence):
        with pytest.raises(ValidationError):
            make_classification(confidence=confidence)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 0.5])
    def test_confidence_boundary_values_accepted(self, confidence):
        result = make_classification(confidence=confidence)
        assert result.confidence == confidence

    def test_empty_summary_rejected(self):
        with pytest.raises(ValidationError):
            make_classification(summary="")

    def test_empty_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            make_classification(reasoning="")

    def test_serialization_round_trip(self):
        result = make_classification(risk_flags=[RiskFlag.SCAM])
        restored = ClassificationResult.model_validate_json(result.model_dump_json())
        assert restored == result
        # enums must serialize to their fixed string values, not arbitrary text
        assert '"category":"NOTIFY"' in result.model_dump_json().replace(" ", "")


# --- Decision ------------------------------------------------------------


class TestDecision:
    def test_valid_construction(self):
        decision = make_decision()
        assert decision.category == Category.NOTIFY
        assert decision.source == DecisionSource.AI_CLASSIFICATION
        assert decision.classification.message_id == decision.message_id

    def test_preference_override_source(self):
        classification = make_classification(category=Category.DIGEST)
        decision = make_decision(
            classification=classification,
            category=Category.NOTIFY,
            reasoning="Sender is a protected contact; forced NOTIFY.",
            source=DecisionSource.PREFERENCE_OVERRIDE,
        )
        assert decision.category == Category.NOTIFY
        assert decision.classification.category == Category.DIGEST
        assert decision.source == DecisionSource.PREFERENCE_OVERRIDE

    def test_message_id_mismatch_with_classification_rejected(self):
        classification = make_classification(message_id="msg-1")
        with pytest.raises(ValidationError):
            make_decision(classification=classification, message_id="msg-DIFFERENT")

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            make_decision(category="NOT_A_CATEGORY")

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            make_decision(source="SOME_OTHER_SOURCE")

    def test_missing_required_field_rejected(self):
        classification = make_classification()
        with pytest.raises(ValidationError):
            Decision(
                message_id=classification.message_id,
                category=Category.NOTIFY,
                classification=classification,
                # missing reasoning and source
            )

    def test_serialization_round_trip(self):
        decision = make_decision()
        restored = Decision.model_validate_json(decision.model_dump_json())
        assert restored == decision
