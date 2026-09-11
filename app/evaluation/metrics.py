"""Pure metric computation: confusion matrices, precision/recall/F1, and
accuracy. No I/O, no Gemini, no repository access — every function here
is a deterministic function of a list of (expected, actual) category
pairs, which is what makes this module trivially and fully unit-testable.

`ClassificationMetrics` and `DecisionMetrics` are intentionally separate
types with an identical shape, computed by the same internal math — this
keeps the two evaluation axes (what the AI understood vs. what the system
finally did) from ever being accidentally interchanged by a type checker
or a careless caller, per the project's core CLASSIFICATION-vs-DECISION
distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.models import Category

CATEGORIES: tuple[Category, ...] = (
    Category.NOTIFY,
    Category.DIGEST,
    Category.MUTE,
    Category.QUARANTINE,
)


@dataclass(frozen=True)
class ConfusionMatrix:
    """`matrix[actual][predicted] = count`, always fully populated over
    all four categories (zero-filled), even when some category never
    appears in the data.
    """

    matrix: dict[Category, dict[Category, int]]

    @classmethod
    def from_pairs(cls, pairs: list[tuple[Category, Category]]) -> ConfusionMatrix:
        matrix = {actual: {predicted: 0 for predicted in CATEGORIES} for actual in CATEGORIES}
        for actual, predicted in pairs:
            matrix[actual][predicted] += 1
        return cls(matrix=matrix)

    def as_table(self) -> list[list[int]]:
        """Row-major grid, rows=actual and cols=predicted, in `CATEGORIES`
        order — convenient for rendering as a plain table.
        """
        return [[self.matrix[actual][predicted] for predicted in CATEGORIES] for actual in CATEGORIES]


@dataclass(frozen=True)
class CategoryMetrics:
    """Precision/recall/F1/support for one category.

    `support` is the number of ground-truth (expected) examples of this
    category — the standard definition. Precision/recall default to 0.0
    when their denominator is zero (no predictions or no ground-truth
    examples of that category) rather than raising.
    """

    category: Category
    precision: float
    recall: float
    f1: float
    support: int


def _compute_per_category(confusion: ConfusionMatrix) -> dict[Category, CategoryMetrics]:
    per_category = {}
    for category in CATEGORIES:
        tp = confusion.matrix[category][category]
        fp = sum(confusion.matrix[actual][category] for actual in CATEGORIES if actual != category)
        fn = sum(confusion.matrix[category][predicted] for predicted in CATEGORIES if predicted != category)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_category[category] = CategoryMetrics(
            category=category, precision=precision, recall=recall, f1=f1, support=support
        )
    return per_category


def _common_fields(pairs: list[tuple[Category, Category]]) -> dict:
    confusion = ConfusionMatrix.from_pairs(pairs)
    total = len(pairs)
    correct = sum(1 for expected, actual in pairs if expected == actual)
    accuracy = correct / total if total > 0 else 0.0

    per_category = _compute_per_category(confusion)
    precisions = [m.precision for m in per_category.values()]
    recalls = [m.recall for m in per_category.values()]
    f1s = [m.f1 for m in per_category.values()]
    # Macro-average (unweighted mean across the 4 categories): with only
    # four, evenly-important categories, macro-averaging treats a miss on
    # a rare category (e.g. QUARANTINE) as seriously as one on a common
    # category — a support-weighted average would understate exactly the
    # safety-critical categories this project cares most about.
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_category": per_category,
        "confusion_matrix": confusion,
    }


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics comparing `ClassificationResult.category` (what the AI
    understood the email to be) against the dataset's expected_category.
    """

    accuracy: float
    correct: int
    total: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_category: dict[Category, CategoryMetrics]
    confusion_matrix: ConfusionMatrix


@dataclass(frozen=True)
class DecisionMetrics:
    """Metrics comparing `Decision.category` (what the system finally did
    for this user) against a deterministically derived expected decision.
    Structurally identical to `ClassificationMetrics` but kept as a
    distinct type — see module docstring.
    """

    accuracy: float
    correct: int
    total: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_category: dict[Category, CategoryMetrics]
    confusion_matrix: ConfusionMatrix


def compute_classification_metrics(pairs: list[tuple[Category, Category]]) -> ClassificationMetrics:
    """`pairs`: list of (expected_category, actual_classification_category)."""
    return ClassificationMetrics(**_common_fields(pairs))


def compute_decision_metrics(pairs: list[tuple[Category, Category]]) -> DecisionMetrics:
    """`pairs`: list of (expected_decision_category, actual_decision_category)."""
    return DecisionMetrics(**_common_fields(pairs))
