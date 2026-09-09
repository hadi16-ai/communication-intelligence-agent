"""Core domain models.

Keeps the distinction between CLASSIFICATION (what an email is, as judged by
the AI) and DECISION (what should happen to it for this particular user)
explicit in the type system rather than as a comment: `ClassificationResult`
and `Decision` are separate models, and only `Decision.category` is the
final, personalized routing outcome the rest of the application should act
on.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    """The four possible routing outcomes for an email."""

    NOTIFY = "NOTIFY"
    DIGEST = "DIGEST"
    MUTE = "MUTE"
    QUARANTINE = "QUARANTINE"


class UrgencyLevel(str, Enum):
    """How time-sensitive an email appears to be."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskFlag(str, Enum):
    """Specific risk indicators an email may raise.

    Kept as a fixed vocabulary (rather than free-form strings) so the
    decision engine can later branch on specific risks (e.g. force
    QUARANTINE on PHISHING) instead of pattern-matching AI text.
    """

    PHISHING = "PHISHING"
    SUSPICIOUS_LINK = "SUSPICIOUS_LINK"
    SPOOFED_SENDER = "SPOOFED_SENDER"
    MALWARE_RISK = "MALWARE_RISK"
    IMPERSONATION = "IMPERSONATION"
    SCAM = "SCAM"
    SUSPICIOUS_ATTACHMENT = "SUSPICIOUS_ATTACHMENT"
    OTHER = "OTHER"


class DecisionSource(str, Enum):
    """Whether the final decision matched the AI's own suggestion or was
    overridden by a deterministic user-preference rule.
    """

    AI_CLASSIFICATION = "AI_CLASSIFICATION"
    PREFERENCE_OVERRIDE = "PREFERENCE_OVERRIDE"


class EmailMessage(BaseModel):
    """A parsed email, independent of its source (Gmail or a synthetic
    fixture) — the shared input to the AI classifier.
    """

    message_id: str = Field(..., min_length=1)
    thread_id: str | None = None
    sender: str = Field(..., min_length=1)
    subject: str = ""
    body: str = ""
    received_at: datetime


class ClassificationResult(BaseModel):
    """What the AI understood about an email — CLASSIFICATION, not DECISION.

    Produced by the AI layer (app/ai) only. `category` here is the model's
    own suggestion and must not be treated as final by the rest of the
    application; it is subject to deterministic override in
    app/core/decisions.py.
    """

    message_id: str = Field(..., min_length=1)
    category: Category
    urgency: UrgencyLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    summary: str = Field(..., min_length=1, max_length=500)
    reasoning: str = Field(..., min_length=1, max_length=1000)


class Decision(BaseModel):
    """The final, personalized routing decision for an email.

    This is the only thing storage and the dashboard should act on — never
    `ClassificationResult.category` directly.
    """

    message_id: str = Field(..., min_length=1)
    category: Category
    reasoning: str = Field(..., min_length=1, max_length=1000)
    source: DecisionSource
    classification: ClassificationResult

    @model_validator(mode="after")
    def _message_id_matches_classification(self) -> Decision:
        if self.message_id != self.classification.message_id:
            raise ValueError(
                "Decision.message_id must match classification.message_id"
            )
        return self
