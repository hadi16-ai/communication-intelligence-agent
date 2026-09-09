"""Structured, deterministic user preferences and matching logic.

This module intentionally does NOT decide what should happen to an email —
it only answers factual questions such as "does this email match the
user's notify keywords?" or "is this sender protected?". Combining these
signals with the AI's classification into a final, personalized `Decision`
is app/core/decisions.py's job (M5). No LLM calls happen here; every method
in this module is a pure, deterministic function of its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.models import EmailMessage


class RuleSet(BaseModel):
    """Keyword and sender patterns associated with one category.

    Matching is a case-insensitive substring check: keywords against the
    email's subject+body, senders against the email's sender string.
    Deliberately simple for Version 1 — Version 2 can extend this shape
    (e.g. per-sender weights, learned patterns, feedback-adjusted rules)
    without changing how it's consumed here.
    """

    keywords: list[str] = Field(default_factory=list)
    senders: list[str] = Field(default_factory=list)


class Preferences(BaseModel):
    """A user's structured communication preferences.

    Kept as plain, versionable data (data/preferences.json) rather than
    logic scattered across the codebase, so preferences can be edited
    directly and — in Version 2 — extended with sender-specific
    preferences, topic preferences, user corrections, feedback history,
    and learned preferences without touching application code.
    """

    protected_senders: list[str] = Field(default_factory=list)
    important_topics: list[str] = Field(default_factory=list)
    notify_rules: RuleSet = Field(default_factory=RuleSet)
    digest_rules: RuleSet = Field(default_factory=RuleSet)
    mute_rules: RuleSet = Field(default_factory=RuleSet)
    quarantine_rules: RuleSet = Field(default_factory=RuleSet)


class RuleMatch(BaseModel):
    """The result of checking an email against a single RuleSet."""

    matched: bool
    matched_keywords: list[str] = Field(default_factory=list)
    matched_sender: bool = False


def _contains_case_insensitive(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


class PreferenceManager:
    """Loads a user's `Preferences` and answers deterministic matching
    questions about a given `EmailMessage`.

    Contains no AI calls and makes no routing decision itself — see
    app/core/decisions.py for how these signals are combined with the AI's
    classification into a final `Decision`.
    """

    def __init__(self, preferences: Preferences):
        self._preferences = preferences

    @classmethod
    def from_file(cls, path: str | Path) -> "PreferenceManager":
        """Load and validate preferences from a JSON file.

        Raises `pydantic.ValidationError` if the file's structure doesn't
        match `Preferences` (e.g. wrong types), and `json.JSONDecodeError`
        if the file isn't valid JSON at all.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Preferences.model_validate(data))

    @property
    def preferences(self) -> Preferences:
        return self._preferences

    def is_protected_sender(self, sender: str) -> bool:
        """Whether `sender` matches any protected-sender pattern —
        independent of the email's content.
        """
        return any(
            _contains_case_insensitive(sender, pattern)
            for pattern in self._preferences.protected_senders
        )

    def matched_important_topics(self, email: EmailMessage) -> list[str]:
        """Which of the user's cross-cutting important topics appear in
        this email's subject/body, if any.
        """
        text = f"{email.subject} {email.body}"
        return [
            topic
            for topic in self._preferences.important_topics
            if _contains_case_insensitive(text, topic)
        ]

    def matches_notify(self, email: EmailMessage) -> RuleMatch:
        return self._match_rule_set(email, self._preferences.notify_rules)

    def matches_digest(self, email: EmailMessage) -> RuleMatch:
        return self._match_rule_set(email, self._preferences.digest_rules)

    def matches_mute(self, email: EmailMessage) -> RuleMatch:
        return self._match_rule_set(email, self._preferences.mute_rules)

    def matches_quarantine(self, email: EmailMessage) -> RuleMatch:
        return self._match_rule_set(email, self._preferences.quarantine_rules)

    def _match_rule_set(self, email: EmailMessage, rule_set: RuleSet) -> RuleMatch:
        text = f"{email.subject} {email.body}"
        matched_keywords = [
            keyword
            for keyword in rule_set.keywords
            if _contains_case_insensitive(text, keyword)
        ]
        matched_sender = any(
            _contains_case_insensitive(email.sender, pattern)
            for pattern in rule_set.senders
        )
        return RuleMatch(
            matched=bool(matched_keywords) or matched_sender,
            matched_keywords=matched_keywords,
            matched_sender=matched_sender,
        )
