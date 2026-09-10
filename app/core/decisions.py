"""Deterministic, personalized decision engine.

CLASSIFICATION (`ClassificationResult`, produced by app/ai in M4) answers
"what is this email?" DECISION (this module) answers "what should happen to
it for this particular user?" The two are never conflated: a `Decision` is
always derived from a classification, but the classification's own
`category` is only ever a starting point — never automatically the final
answer.

This module makes no network calls, calls no LLM, and is fully
deterministic: the same (email, classification, preferences) always
produces the same `Decision`.

Decision priority (highest to lowest):

    1. SECURITY    Meaningful risk — either the classification itself
                    (`risk_flags` or `category == QUARANTINE`) or an
                    explicit user quarantine rule — forces QUARANTINE,
                    unconditionally. Nothing below this point can override
                    it: not a protected sender, not "urgent" wording, not
                    a claimed bank/manager identity, not text inside the
                    email instructing otherwise.
    2. PREFERENCE  Explicit, personalized rules decide NOTIFY / MUTE /
                    DIGEST when they clearly apply: protected senders,
                    important topics, and the notify/mute/digest keyword
                    and sender rules from app/preferences.
    3. CATEGORY    No preference rule matched. Defers to the
                    classification's own category, escalated to NOTIFY if
                    the classification marked the email HIGH urgency
                    (never silently downgraded).
    4. FALLBACK    Covered by step 3 (deferring to classification is
                    itself the safe fallback) — see `_fallback_decision`.

This ordering exists because security must never depend on personalization
being configured correctly, and personalization must never depend on the
AI's own (unverified) category once a human-authored rule already answers
the question.
"""

from __future__ import annotations

from app.core.models import Category, ClassificationResult, Decision, DecisionSource, EmailMessage, UrgencyLevel
from app.preferences.manager import PreferenceManager, RuleMatch


def _describe_match(match: RuleMatch) -> str:
    parts = []
    if match.matched_keywords:
        parts.append(f"keywords: {', '.join(match.matched_keywords)}")
    if match.matched_sender:
        parts.append("sender pattern")
    return "; ".join(parts) if parts else "matched"


class DecisionEngine:
    """Combines a classification with the user's preferences to produce a
    final, personalized, explainable `Decision`.

    Stateless and side-effect free: every method is a pure function of its
    arguments, so the same inputs always produce the same output.
    """

    def decide(
        self,
        email: EmailMessage,
        classification: ClassificationResult,
        preferences: PreferenceManager,
    ) -> Decision:
        if classification.message_id != email.message_id:
            raise ValueError(
                f"classification.message_id ({classification.message_id!r}) does not "
                f"match email.message_id ({email.message_id!r}); refusing to build a "
                "Decision from mismatched inputs."
            )

        security = self._check_security(email, classification, preferences)
        if security is not None:
            category, source, reason = security
            return self._build(email, classification, category, source, reason)

        preference = self._check_preferences(email, classification, preferences)
        if preference is not None:
            category, source, reason = preference
            return self._build(email, classification, category, source, reason)

        category, source, reason = self._fallback_decision(classification)
        return self._build(email, classification, category, source, reason)

    # --- Priority 1: security -------------------------------------------

    def _check_security(
        self,
        email: EmailMessage,
        classification: ClassificationResult,
        preferences: PreferenceManager,
    ) -> tuple[Category, DecisionSource, str] | None:
        """Unconditional QUARANTINE check. Nothing after this method runs
        can change the outcome if it returns a result — this is the one
        rule in the engine that a protected sender, urgent wording, or
        text embedded in the email itself must never be able to bypass.
        """
        if classification.risk_flags:
            flags = ", ".join(flag.value for flag in classification.risk_flags)
            return (
                Category.QUARANTINE,
                DecisionSource.AI_CLASSIFICATION,
                f"Quarantine: classification identified security risk indicators ({flags}).",
            )

        if classification.category == Category.QUARANTINE:
            return (
                Category.QUARANTINE,
                DecisionSource.AI_CLASSIFICATION,
                "Quarantine: classification identified this email as a security risk.",
            )

        quarantine_match = preferences.matches_quarantine(email)
        if quarantine_match.matched:
            return (
                Category.QUARANTINE,
                DecisionSource.PREFERENCE_OVERRIDE,
                f"Quarantine: matches a user-defined quarantine rule ({_describe_match(quarantine_match)}).",
            )

        return None

    # --- Priority 2: personalized preferences ----------------------------

    def _check_preferences(
        self,
        email: EmailMessage,
        classification: ClassificationResult,
        preferences: PreferenceManager,
    ) -> tuple[Category, DecisionSource, str] | None:
        """Only reached once security has cleared the email. A protected
        sender alone does not force NOTIFY here — it only counts alongside
        elevated urgency, matching "do not blindly trust a sender merely
        because it matches a preference."
        """
        protected = preferences.is_protected_sender(email.sender)
        important_topics = preferences.matched_important_topics(email)
        notify_match = preferences.matches_notify(email)

        notify_reasons = []
        if notify_match.matched:
            notify_reasons.append(f"matches notify rule ({_describe_match(notify_match)})")
        if important_topics:
            notify_reasons.append(f"matches important topic(s): {', '.join(important_topics)}")
        if protected and classification.urgency in (UrgencyLevel.MEDIUM, UrgencyLevel.HIGH):
            notify_reasons.append("sender is protected and urgency is elevated")

        if notify_reasons:
            return (
                Category.NOTIFY,
                DecisionSource.PREFERENCE_OVERRIDE,
                "Notify: " + "; ".join(notify_reasons) + ".",
            )

        # Mute is checked before digest: an explicit "this is
        # advertising/marketing" signal is more specific than a general
        # "this topic is fine for later" signal. HIGH urgency still wins
        # over mute — the classifier flagging genuine urgency outranks a
        # keyword match that merely looks promotional.
        mute_match = preferences.matches_mute(email)
        if mute_match.matched and classification.urgency != UrgencyLevel.HIGH:
            return (
                Category.MUTE,
                DecisionSource.PREFERENCE_OVERRIDE,
                f"Mute: matches user mute preferences ({_describe_match(mute_match)}).",
            )

        digest_match = preferences.matches_digest(email)
        if digest_match.matched:
            return (
                Category.DIGEST,
                DecisionSource.PREFERENCE_OVERRIDE,
                f"Digest: matches user digest preferences ({_describe_match(digest_match)}).",
            )

        return None

    # --- Priority 3/4: category + urgency fallback -----------------------

    def _fallback_decision(
        self, classification: ClassificationResult
    ) -> tuple[Category, DecisionSource, str]:
        """No explicit preference rule applied. Defers to the
        classification's own category rather than assuming MUTE, but a
        HIGH urgency classification always escalates to NOTIFY — an email
        the classifier considers urgent is never silently downgraded just
        because no preference rule happened to match it.

        `classification.category` can only be NOTIFY, DIGEST, or MUTE by
        this point: QUARANTINE is always caught by `_check_security`.
        """
        if classification.urgency == UrgencyLevel.HIGH:
            return (
                Category.NOTIFY,
                DecisionSource.AI_CLASSIFICATION,
                "Notify: classification indicates high urgency and no preference rule applied.",
            )

        return (
            classification.category,
            DecisionSource.AI_CLASSIFICATION,
            f"{classification.category.value.title()}: no preference rule matched; "
            f"deferring to classification (urgency: {classification.urgency.value}).",
        )

    # --- Construction ------------------------------------------------------

    def _build(
        self,
        email: EmailMessage,
        classification: ClassificationResult,
        category: Category,
        source: DecisionSource,
        reason: str,
    ) -> Decision:
        return Decision(
            message_id=email.message_id,
            category=category,
            reasoning=reason,
            source=source,
            classification=classification,
        )
