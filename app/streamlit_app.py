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

from pathlib import Path

import streamlit as st

from app.core.config import Settings
from app.core.decisions import DecisionEngine
from app.core.models import Category, DecisionSource
from app.evaluation.evaluator import EvaluationReport, evaluate
from app.preferences.manager import PreferenceManager
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
from app.ui.evaluation_view import (
    category_metrics_rows,
    confusion_matrix_rows,
    coverage_message,
    format_recall,
    mismatch_rows,
)

st.set_page_config(
    page_title="Communication Intelligence Agent",
    page_icon="📬",
    layout="wide",
)

PREFERENCES_PATH = Path(__file__).resolve().parent.parent / "data" / "preferences.json"


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


def _get_preferences() -> PreferenceManager:
    return PreferenceManager.from_file(PREFERENCES_PATH)


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


def render_inbox_tab(repository: EmailRepository) -> None:
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


def render_evaluation_tab(repository: EmailRepository) -> None:
    """Read-only evaluation of stored classifications/decisions against
    data/eval_dataset.json ground truth. Computes nothing itself beyond
    calling app.evaluation.evaluate() — no Gemini call, no metric math
    lives in this file (see app/evaluation and app/ui/evaluation_view.py).
    """
    preferences = _get_preferences()
    report: EvaluationReport = evaluate(repository, preferences, DecisionEngine())
    coverage = report.coverage

    message = coverage_message(report)
    if coverage.is_partial:
        st.warning(f"⚠️ {message}")
        if coverage.missing_message_ids:
            with st.expander(f"{coverage.missing_count} email(s) not yet evaluated"):
                st.write(", ".join(coverage.missing_message_ids))
    else:
        st.success(message)

    st.subheader("Overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Evaluated", coverage.evaluated_count)
    overview_cols[1].metric("Total in dataset", coverage.total_count)
    overview_cols[2].metric("Coverage", f"{coverage.coverage_percent:.1f}%")
    overview_cols[3].metric("Classification accuracy", f"{report.classification.accuracy * 100:.1f}%")

    if coverage.evaluated_count == 0:
        st.info("Nothing has been evaluated yet — no stored classifications match the dataset.")
        return

    st.divider()
    st.subheader("🛡️ Safety-critical metrics")
    st.caption(
        "Missing a genuinely important job/interview/security email is more costly than an "
        "extra notification; failing to quarantine dangerous content is separately safety-critical."
    )
    safety_cols = st.columns(4)
    safety_cols[0].metric(
        f"{CATEGORY_EMOJI[Category.NOTIFY]} NOTIFY recall (classification)",
        format_recall(report.classification.per_category[Category.NOTIFY]),
    )
    safety_cols[1].metric(
        f"{CATEGORY_EMOJI[Category.NOTIFY]} NOTIFY recall (decision)",
        format_recall(report.decision.per_category[Category.NOTIFY]),
    )
    safety_cols[2].metric(
        f"{CATEGORY_EMOJI[Category.QUARANTINE]} QUARANTINE recall (classification)",
        format_recall(report.classification.per_category[Category.QUARANTINE]),
    )
    safety_cols[3].metric(
        f"{CATEGORY_EMOJI[Category.QUARANTINE]} QUARANTINE recall (decision)",
        format_recall(report.decision.per_category[Category.QUARANTINE]),
    )

    st.divider()
    class_col, decision_col = st.columns(2)

    with class_col:
        st.markdown("#### Classification metrics")
        st.caption("AI-understood category vs. dataset expected_category")
        st.dataframe(category_metrics_rows(report.classification.per_category), hide_index=True, width="stretch")
        st.caption(
            f"Macro precision {report.classification.macro_precision:.3f} · "
            f"macro recall {report.classification.macro_recall:.3f} · "
            f"macro F1 {report.classification.macro_f1:.3f}"
        )
        st.markdown("**Confusion matrix — classification**")
        st.dataframe(confusion_matrix_rows(report.classification.confusion_matrix), hide_index=True, width="stretch")

    with decision_col:
        st.markdown("#### Decision metrics")
        st.caption("Final decision vs. expected outcome of the existing decision policy")
        st.dataframe(category_metrics_rows(report.decision.per_category), hide_index=True, width="stretch")
        st.caption(
            f"Macro precision {report.decision.macro_precision:.3f} · "
            f"macro recall {report.decision.macro_recall:.3f} · "
            f"macro F1 {report.decision.macro_f1:.3f}"
        )
        st.markdown("**Confusion matrix — decision**")
        st.dataframe(confusion_matrix_rows(report.decision.confusion_matrix), hide_index=True, width="stretch")

    if report.classification_mismatches:
        with st.expander(f"Classification mismatches ({len(report.classification_mismatches)})"):
            st.dataframe(mismatch_rows(report.classification_mismatches), hide_index=True, width="stretch")

    if report.decision_mismatches:
        with st.expander(f"Decision mismatches ({len(report.decision_mismatches)})"):
            st.dataframe(mismatch_rows(report.decision_mismatches), hide_index=True, width="stretch")


def main() -> None:
    render_header()

    repository = _get_repository()

    inbox_tab, evaluation_tab = st.tabs(["📥 Inbox", "📊 Evaluation"])
    with inbox_tab:
        render_inbox_tab(repository)
    with evaluation_tab:
        render_evaluation_tab(repository)


main()
