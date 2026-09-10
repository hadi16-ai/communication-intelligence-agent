"""Pure data-preparation logic for the Streamlit dashboard.

Deliberately free of any `streamlit` import so it can be unit-tested
directly without spinning up the app. Reuses the existing
`EmailRepository` for all data access and introduces no new persistence
or decision-making logic of its own — classification and routing stay in
app.ai / app.core.decisions exactly as before; this module only shapes
already-stored data for display.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.models import Category, Decision, DecisionSource, RiskFlag
from app.storage.repositories import EmailRepository, StoredEmailRecord

CATEGORY_EMOJI: dict[Category, str] = {
    Category.NOTIFY: "🔴",
    Category.DIGEST: "🟡",
    Category.MUTE: "⚪",
    Category.QUARANTINE: "🛡️",
}


@dataclass(frozen=True)
class DashboardSummary:
    """Counts of stored records by their FINAL DECISION category.

    Deliberately counts `Decision.category`, never
    `ClassificationResult.category` — the decision is the only thing the
    rest of the app treats as "what actually happened" to an email.
    Records with a classification but no decision yet are included in
    `total` but not in any category count.
    """

    notify: int
    digest: int
    mute: int
    quarantine: int
    total: int

    @property
    def pending(self) -> int:
        """Records classified but not yet decided (decision is None)."""
        return self.total - (self.notify + self.digest + self.mute + self.quarantine)


def load_records(repository: EmailRepository) -> list[StoredEmailRecord]:
    """All stored records. Thin pass-through kept so the dashboard code
    depends on this module rather than reaching into the repository
    layer directly in more than one place.
    """
    return repository.list_records()


def summarize(records: list[StoredEmailRecord]) -> DashboardSummary:
    counts = {category: 0 for category in Category}
    for record in records:
        if record.decision is not None:
            counts[record.decision.category] += 1
    return DashboardSummary(
        notify=counts[Category.NOTIFY],
        digest=counts[Category.DIGEST],
        mute=counts[Category.MUTE],
        quarantine=counts[Category.QUARANTINE],
        total=len(records),
    )


def filter_records(
    records: list[StoredEmailRecord],
    category: Category | None = None,
    search: str = "",
) -> list[StoredEmailRecord]:
    """Filters by final decision category (None = all categories) and an
    optional case-insensitive substring search over sender + subject.

    A record with no decision yet is excluded by any specific category
    filter (it belongs to none of the four categories) but is always
    included when `category` is None.
    """
    result = records
    if category is not None:
        result = [r for r in result if r.decision is not None and r.decision.category == category]
    if search:
        needle = search.strip().lower()
        if needle:
            result = [r for r in result if needle in r.sender.lower() or needle in r.subject.lower()]
    return result


def format_risk_flags(risk_flags: list[RiskFlag]) -> str:
    return ", ".join(flag.value for flag in risk_flags) if risk_flags else "—"


def decision_status_label(record: StoredEmailRecord) -> str:
    """What to show in a "decision" column/field when a record hasn't
    been decided yet — an honest state, not an invented category.
    """
    if record.decision is None:
        return "PENDING"
    return record.decision.category.value


def decision_trace_label(decision: Decision) -> str:
    """A short, honest label for how a decision was reached — derived only
    from fields the DecisionEngine actually populated (`source`), never by
    re-evaluating preferences against the CURRENT preferences.json (which
    could have changed since the decision was made, and would risk
    displaying a preference "match" that was never actually evaluated for
    this stored decision).
    """
    if decision.source == DecisionSource.PREFERENCE_OVERRIDE:
        return "Matched a personalized preference rule"
    return "Followed the AI classification directly (no preference rule applied)"


def to_table_row(record: StoredEmailRecord) -> dict:
    """One row for the main email table. Never includes the raw email
    body — it was never stored in SQLite in the first place (see
    app/storage/database.py).
    """
    return {
        "message_id": record.message_id,
        "sender": record.sender,
        "subject": record.subject,
        "received_at": record.received_at.isoformat(sep=" ", timespec="minutes"),
        "decision": decision_status_label(record),
        "urgency": record.classification.urgency.value,
        "confidence": record.classification.confidence,
        "risk_flags": format_risk_flags(record.classification.risk_flags),
    }
