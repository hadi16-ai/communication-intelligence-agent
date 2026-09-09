"""Tests for app.preferences.manager: Preferences, RuleSet, RuleMatch, and
PreferenceManager. No AI calls — every check here is deterministic.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.models import EmailMessage
from app.preferences.manager import PreferenceManager, Preferences, RuleSet

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_PREFERENCES_PATH = REPO_ROOT / "data" / "preferences.json"


def make_email(**overrides) -> EmailMessage:
    defaults = dict(
        message_id="msg-1",
        sender="someone@example.com",
        subject="Just checking in",
        body="Hope you're doing well.",
        received_at=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


def make_test_preferences() -> Preferences:
    """A small, self-contained preference set for unit tests — independent
    of the shipped data/preferences.json content so these tests don't break
    if that example file is edited.
    """
    return Preferences(
        protected_senders=["manager@mycompany.com", "@mybank.com"],
        important_topics=["job", "interview", "deadline"],
        notify_rules=RuleSet(
            keywords=["interview", "recruiter", "hiring manager", "deadline"],
            senders=["@mycompany.com", "@greenhouse.io"],
        ),
        digest_rules=RuleSet(
            keywords=["ai agent", "new model", "hackathon", "family"],
            senders=["@substack.com"],
        ),
        mute_rules=RuleSet(
            keywords=["buy now", "discount", "% off", "unsubscribe"],
            senders=["@promotions."],
        ),
        quarantine_rules=RuleSet(
            keywords=["wire transfer", "gift card", "verify your password"],
            senders=[],
        ),
    )


@pytest.fixture
def manager() -> PreferenceManager:
    return PreferenceManager(make_test_preferences())


# --- Loading ---------------------------------------------------------


class TestLoading:
    def test_loading_valid_preferences_from_file(self, tmp_path):
        path = tmp_path / "preferences.json"
        path.write_text(make_test_preferences().model_dump_json(), encoding="utf-8")

        loaded = PreferenceManager.from_file(path)

        assert loaded.preferences == make_test_preferences()

    def test_shipped_preferences_file_loads_and_is_valid(self):
        loaded = PreferenceManager.from_file(SHIPPED_PREFERENCES_PATH)

        assert loaded.preferences.protected_senders
        assert loaded.preferences.notify_rules.keywords
        assert loaded.preferences.digest_rules.keywords
        assert loaded.preferences.mute_rules.keywords
        assert loaded.preferences.quarantine_rules.keywords

    def test_missing_fields_default_to_empty(self):
        prefs = Preferences.model_validate({"protected_senders": ["x@y.com"]})

        assert prefs.protected_senders == ["x@y.com"]
        assert prefs.important_topics == []
        assert prefs.notify_rules == RuleSet()
        assert prefs.digest_rules.keywords == []

    def test_empty_object_is_valid_and_fully_empty(self):
        prefs = Preferences.model_validate({})

        assert prefs.protected_senders == []
        assert prefs.notify_rules.keywords == []
        assert prefs.quarantine_rules.senders == []

    def test_malformed_protected_senders_type_rejected(self):
        with pytest.raises(ValidationError):
            Preferences.model_validate({"protected_senders": "not-a-list"})

    def test_malformed_rule_set_type_rejected(self):
        with pytest.raises(ValidationError):
            Preferences.model_validate({"notify_rules": ["not", "an", "object"]})

    def test_malformed_keywords_type_within_rule_set_rejected(self):
        with pytest.raises(ValidationError):
            Preferences.model_validate({"notify_rules": {"keywords": "not-a-list"}})

    def test_invalid_json_file_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(Exception):
            PreferenceManager.from_file(path)


# --- Protected senders / important topics -----------------------------


class TestProtectedSendersAndTopics:
    def test_is_protected_sender_matches(self, manager):
        assert manager.is_protected_sender("manager@mycompany.com") is True

    def test_is_protected_sender_matches_domain_substring(self, manager):
        assert manager.is_protected_sender("alerts@mybank.com") is True

    def test_is_protected_sender_no_match(self, manager):
        assert manager.is_protected_sender("stranger@unknown.com") is False

    def test_is_protected_sender_case_insensitive(self, manager):
        assert manager.is_protected_sender("MANAGER@MYCOMPANY.COM") is True

    def test_matched_important_topics(self, manager):
        email = make_email(subject="Interview next week", body="Please confirm the deadline.")

        matched = manager.matched_important_topics(email)

        assert set(matched) == {"interview", "deadline"}

    def test_matched_important_topics_no_match(self, manager):
        email = make_email(subject="Hello", body="Just saying hi.")

        assert manager.matched_important_topics(email) == []


# --- NOTIFY: job-related / interview / company ------------------------


class TestNotifyMatching:
    def test_matches_job_related_email_by_keyword(self, manager):
        email = make_email(
            subject="Update on your application",
            body="Our hiring manager would like to schedule a call.",
        )

        result = manager.matches_notify(email)

        assert result.matched is True
        assert "hiring manager" in result.matched_keywords

    def test_matches_interview_email_by_keyword(self, manager):
        email = make_email(subject="Interview confirmation", body="See you Thursday at 2pm.")

        result = manager.matches_notify(email)

        assert result.matched is True
        assert "interview" in result.matched_keywords

    def test_matches_company_email_by_sender_domain(self, manager):
        email = make_email(
            sender="hr@mycompany.com",
            subject="Team update",
            body="Nothing urgent, just an FYI.",
        )

        result = manager.matches_notify(email)

        assert result.matched is True
        assert result.matched_sender is True
        assert result.matched_keywords == []

    def test_no_match_for_unrelated_email(self, manager):
        email = make_email(subject="Weekend plans", body="Want to grab lunch?")

        result = manager.matches_notify(email)

        assert result.matched is False
        assert result.matched_keywords == []
        assert result.matched_sender is False


# --- DIGEST: AI/tech news, hackathons, friends/family ------------------


class TestDigestMatching:
    def test_matches_ai_news_topic(self, manager):
        email = make_email(
            subject="This week in AI",
            body="A new AI agent framework was released today.",
        )

        result = manager.matches_digest(email)

        assert result.matched is True
        assert "ai agent" in result.matched_keywords

    def test_matches_hackathon_topic(self, manager):
        email = make_email(subject="Hackathon this weekend!", body="Come build something fun.")

        result = manager.matches_digest(email)

        assert result.matched is True
        assert "hackathon" in result.matched_keywords

    def test_matches_newsletter_sender(self, manager):
        email = make_email(
            sender="digest@substack.com",
            subject="Your weekly roundup",
            body="Here's what happened this week.",
        )

        result = manager.matches_digest(email)

        assert result.matched is True
        assert result.matched_sender is True


# --- MUTE: promotional / marketing -------------------------------------


class TestMuteMatching:
    def test_matches_promotional_email(self, manager):
        email = make_email(
            subject="Huge sale - buy now!",
            body="Everything is 50% off this weekend only.",
        )

        result = manager.matches_mute(email)

        assert result.matched is True
        assert "buy now" in result.matched_keywords

    def test_matches_marketing_sender(self, manager):
        email = make_email(
            sender="deals@promotions.retailer.com",
            subject="Don't miss out",
            body="Check out our latest collection.",
        )

        result = manager.matches_mute(email)

        assert result.matched is True
        assert result.matched_sender is True

    def test_no_match_for_normal_email(self, manager):
        email = make_email(subject="Project status", body="Everything is on track.")

        result = manager.matches_mute(email)

        assert result.matched is False


# --- QUARANTINE: scams / phishing --------------------------------------


class TestQuarantineMatching:
    def test_matches_phishing_keyword(self, manager):
        email = make_email(
            subject="Security notice",
            body="Please verify your password immediately or your account will be locked.",
        )

        result = manager.matches_quarantine(email)

        assert result.matched is True
        assert "verify your password" in result.matched_keywords

    def test_matches_scam_keyword(self, manager):
        email = make_email(
            subject="Congratulations!",
            body="You have won a prize. Send a gift card to claim it.",
        )

        result = manager.matches_quarantine(email)

        assert result.matched is True
        assert "gift card" in result.matched_keywords

    def test_no_match_for_legitimate_email(self, manager):
        email = make_email(subject="Invoice attached", body="Please find this month's invoice.")

        result = manager.matches_quarantine(email)

        assert result.matched is False


# --- Case-insensitivity across all rule types ---------------------------


class TestCaseInsensitivity:
    def test_notify_keyword_matches_regardless_of_case(self, manager):
        email = make_email(subject="INTERVIEW CONFIRMATION", body="")

        assert manager.matches_notify(email).matched is True

    def test_sender_pattern_matches_regardless_of_case(self, manager):
        email = make_email(sender="HR@MyCompany.COM", subject="", body="")

        assert manager.matches_notify(email).matched is True

    def test_quarantine_keyword_matches_mixed_case(self, manager):
        email = make_email(subject="", body="Please Verify Your Password now.")

        assert manager.matches_quarantine(email).matched is True
