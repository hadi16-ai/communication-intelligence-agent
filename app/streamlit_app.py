"""Streamlit dashboard: read-only presentation layer over the existing
SQLite-backed EmailRepository.

Architecture reminder: this file contains NO classification or decision
logic of its own. It only reads what app.ai / app.core.decisions / app.
storage already produced and stored (see app/core/pipeline.py for how
those get populated). It never calls Gemini, never touches Gmail, and
never writes anything back to the database.

EmailMessage -> Classifier -> ClassificationResult -> DecisionEngine ->
Decision -> Repository/SQLite -> (this file reads only, from here on)
"""

from __future__ import annotations

import streamlit as st

from app.core.config import Settings
from app.core.models import Category, DecisionSource
from app.storage.repositories import EmailRepository
from app.ui.dashboard_data import (
    CATEGORY_EMOJI,
    decision_trace_label,
    filter_records,
    format_risk_flags,
    load_records,
    summarize,
    to_table_row,
)

st.set_page_config(
    page_title="Communication Intelligence Agent",
    page_icon="📬",
    layout="wide",
)


def _get_repository() -> EmailRepository:
    """A fresh, cheap repository handle for this script run.

    Deliberately not cached across runs (`EmailRepository` just wraps a
    path — no expensive connection is opened until a method is called):
    caching it would mean a changed `DATABASE_PATH` — or a different
    database entirely — could silently keep serving a stale one.
    `initialize()` is idempotent and safe to call every run — it creates
    the database/schema on first use and never touches existing data.
    """
    settings = Settings()
    repository = EmailRepository(settings.database_path)
    repository.initialize()
    return repository


def render_header() -> None:
    st.title("Communication Intelligence Agent")
    st.caption("Personalized email triage: Notify, Digest, Mute, or Quarantine.")


def render_summary(summary) -> None:
    columns = st.columns(5)
    columns[0].metric(f"{CATEGORY_EMOJI[Category.NOTIFY]} NOTIFY", summary.notify)
    columns[1].metric(f"{CATEGORY_EMOJI[Category.DIGEST]} DIGEST", summary.digest)
    columns[2].metric(f"{CATEGORY_EMOJI[Category.MUTE]} MUTE", summary.mute)
    columns[3].metric(f"{CATEGORY_EMOJI[Category.QUARANTINE]} QUARANTINE", summary.quarantine)
    columns[4].metric("Total processed", summary.total)
    if summary.pending:
        st.caption(f"{summary.pending} record(s) classified but not yet decided.")


def render_filters(records):
    filter_options: list[tuple[str, Category | None]] = [
        ("All", None),
        (f"{CATEGORY_EMOJI[Category.NOTIFY]} NOTIFY", Category.NOTIFY),
        (f"{CATEGORY_EMOJI[Category.DIGEST]} DIGEST", Category.DIGEST),
        (f"{CATEGORY_EMOJI[Category.MUTE]} MUTE", Category.MUTE),
        (f"{CATEGORY_EMOJI[Category.QUARANTINE]} QUARANTINE", Category.QUARANTINE),
    ]

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        label = st.radio(
            "Filter by decision",
            options=[opt[0] for opt in filter_options],
            horizontal=True,
        )
        selected_category = dict(filter_options)[label]
    with search_col:
        search = st.text_input("Search sender or subject", value="")

    return filter_records(records, category=selected_category, search=search)


def render_table(filtered_records) -> None:
    st.subheader(f"Emails ({len(filtered_records)})")
    if not filtered_records:
        st.caption("No emails match the current filter.")
        return
    rows = [to_table_row(r) for r in filtered_records]
    st.dataframe(
        rows,
        column_order=["sender", "subject", "received_at", "decision", "urgency", "confidence", "risk_flags"],
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
        hide_index=True,
        width="stretch",
    )


def render_detail(filtered_records) -> None:
    if not filtered_records:
        return

    st.subheader("Decision trace")
    options = {
        f"{r.subject or '(no subject)'} — {r.sender} [{r.message_id}]": r.message_id
        for r in filtered_records
    }
    selected_label = st.selectbox("Select an email to inspect", options=list(options.keys()))
    selected_id = options[selected_label]
    record = next(r for r in filtered_records if r.message_id == selected_id)

    classification = record.classification
    decision = record.decision

    if decision is not None and decision.category == Category.QUARANTINE:
        st.error("🛡️ QUARANTINE — this message was flagged as a security risk.")
    elif classification.risk_flags:
        st.warning(f"⚠️ Risk indicators present: {format_risk_flags(classification.risk_flags)}")

    st.markdown("**Email → Classification → Preferences → Decision**")

    class_col, decision_col = st.columns(2)

    with class_col:
        st.markdown("#### Classification")
        st.write(f"**Category (AI suggestion):** {classification.category.value}")
        st.write(f"**Urgency:** {classification.urgency.value}")
        st.write(f"**Confidence:** {classification.confidence:.2f}")
        st.progress(classification.confidence)
        st.write(f"**Risk flags:** {format_risk_flags(classification.risk_flags)}")
        st.write("**Summary:**")
        st.caption(classification.summary)
        st.write("**Reasoning:**")
        st.caption(classification.reasoning)

    with decision_col:
        st.markdown("#### Decision")
        if decision is None:
            st.info("This email has been classified but not yet decided.")
        else:
            st.write(f"**Final decision:** {decision.category.value}")
            st.write(f"**Decision source:** {decision.source.value}")
            st.write(f"**{decision_trace_label(decision)}**")
            st.write("**Decision reasoning (includes matched preference rule, if any):**")
            st.caption(decision.reasoning)


def main() -> None:
    render_header()

    repository = _get_repository()
    records = load_records(repository)

    if not records:
        st.info(
            "No processed emails yet. This dashboard only reads existing data — "
            "it never calls Gemini automatically. Run the pipeline "
            "(see scripts/run_synthetic_pipeline.py or the M7 EmailProcessor) "
            "to populate the database first."
        )
        return

    summary = summarize(records)
    render_summary(summary)
    st.divider()

    filtered_records = render_filters(records)
    render_table(filtered_records)
    st.divider()
    render_detail(filtered_records)


main()
