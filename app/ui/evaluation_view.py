"""Presentation-ready shaping of EvaluationReport for the Streamlit
Evaluation tab. Deliberately free of any `streamlit` import so it can be
unit-tested directly. Contains NO metric computation of its own — every
number originates in app.evaluation; this module only formats it.
"""

from __future__ import annotations

from app.evaluation.evaluator import EvaluationReport
from app.evaluation.metrics import CATEGORIES, CategoryMetrics, ConfusionMatrix


def format_recall(metrics: CategoryMetrics) -> str:
    """Formats a recall percentage — but calls out zero support
    explicitly rather than presenting a bare "0.0%", which could be
    misread as "the system failed every example" when the real situation
    is "no examples of this category have been evaluated yet" (exactly
    the current QUARANTINE state: 0 of the 6 cached emails are
    QUARANTINE-labeled). This matters most for the safety-critical
    NOTIFY/QUARANTINE recall figures, which are meant to be read at a
    glance.
    """
    if metrics.support == 0:
        return "N/A (0 examples)"
    return f"{metrics.recall * 100:.1f}%"


def coverage_message(report: EvaluationReport) -> str:
    coverage = report.coverage
    if coverage.total_count == 0:
        return "No evaluation dataset found."
    if not coverage.is_partial:
        return f"Full evaluation — all {coverage.total_count} emails evaluated."
    return (
        f"Partial evaluation — {coverage.evaluated_count} of {coverage.total_count} emails "
        f"evaluated ({coverage.coverage_percent:.1f}% coverage). "
        f"{coverage.missing_count} email(s) have not been classified yet "
        "(no Gemini call was made to fill them in)."
    )


def confusion_matrix_rows(matrix: ConfusionMatrix) -> list[dict]:
    """One row per actual category, one column per predicted category —
    shaped for st.dataframe.
    """
    rows = []
    for actual in CATEGORIES:
        row = {"actual \\ predicted": actual.value}
        for predicted in CATEGORIES:
            row[predicted.value] = matrix.matrix[actual][predicted]
        rows.append(row)
    return rows


def category_metrics_rows(per_category: dict[object, CategoryMetrics]) -> list[dict]:
    return [
        {
            "category": category.value,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "support": metrics.support,
        }
        for category, metrics in per_category.items()
    ]


def mismatch_rows(mismatches) -> list[dict]:
    return [
        {
            "message_id": m.message_id,
            "expected": m.expected.value,
            "actual": m.actual.value,
            "notes": m.notes,
        }
        for m in mismatches
    ]
