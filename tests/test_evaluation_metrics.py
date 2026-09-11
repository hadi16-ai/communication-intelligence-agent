"""Tests for app.evaluation.metrics: ConfusionMatrix, CategoryMetrics,
ClassificationMetrics/DecisionMetrics, and their compute functions.

Pure, deterministic, hand-verified math — no I/O, no Gemini.
"""

import pytest

from app.core.models import Category
from app.evaluation.metrics import (
    CATEGORIES,
    ConfusionMatrix,
    compute_classification_metrics,
    compute_decision_metrics,
)

NOTIFY, DIGEST, MUTE, QUARANTINE = Category.NOTIFY, Category.DIGEST, Category.MUTE, Category.QUARANTINE


class TestConfusionMatrix:
    def test_empty_pairs_gives_a_fully_zeroed_matrix(self):
        matrix = ConfusionMatrix.from_pairs([])

        for actual in CATEGORIES:
            for predicted in CATEGORIES:
                assert matrix.matrix[actual][predicted] == 0

    def test_counts_pairs_correctly(self):
        pairs = [(NOTIFY, NOTIFY), (NOTIFY, DIGEST), (DIGEST, DIGEST)]

        matrix = ConfusionMatrix.from_pairs(pairs)

        assert matrix.matrix[NOTIFY][NOTIFY] == 1
        assert matrix.matrix[NOTIFY][DIGEST] == 1
        assert matrix.matrix[DIGEST][DIGEST] == 1
        assert matrix.matrix[QUARANTINE][QUARANTINE] == 0

    def test_as_table_matches_categories_order(self):
        pairs = [(NOTIFY, NOTIFY)]
        matrix = ConfusionMatrix.from_pairs(pairs)

        table = matrix.as_table()

        assert table[0][0] == 1  # NOTIFY row, NOTIFY column
        assert len(table) == len(CATEGORIES)
        assert all(len(row) == len(CATEGORIES) for row in table)


class TestPerfectPredictions:
    def test_all_correct_gives_accuracy_one(self):
        pairs = [(NOTIFY, NOTIFY), (DIGEST, DIGEST), (MUTE, MUTE), (QUARANTINE, QUARANTINE)]

        metrics = compute_classification_metrics(pairs)

        assert metrics.accuracy == 1.0
        assert metrics.correct == 4
        assert metrics.total == 4
        for category in CATEGORIES:
            m = metrics.per_category[category]
            assert m.precision == 1.0
            assert m.recall == 1.0
            assert m.f1 == 1.0
            assert m.support == 1
        assert metrics.macro_precision == 1.0
        assert metrics.macro_recall == 1.0
        assert metrics.macro_f1 == 1.0


class TestAllIncorrectPredictions:
    def test_all_wrong_gives_accuracy_zero(self):
        pairs = [(NOTIFY, DIGEST), (DIGEST, MUTE), (MUTE, QUARANTINE), (QUARANTINE, NOTIFY)]

        metrics = compute_classification_metrics(pairs)

        assert metrics.accuracy == 0.0
        assert metrics.correct == 0
        for category in CATEGORIES:
            assert metrics.per_category[category].precision == 0.0
            assert metrics.per_category[category].recall == 0.0
            assert metrics.per_category[category].f1 == 0.0


class TestZeroSupportCategory:
    def test_category_never_appearing_gets_zeroed_metrics_not_an_error(self):
        pairs = [(NOTIFY, NOTIFY), (DIGEST, DIGEST)]  # MUTE, QUARANTINE never appear

        metrics = compute_classification_metrics(pairs)

        assert metrics.per_category[MUTE].support == 0
        assert metrics.per_category[MUTE].precision == 0.0
        assert metrics.per_category[MUTE].recall == 0.0
        assert metrics.per_category[MUTE].f1 == 0.0
        assert metrics.per_category[QUARANTINE].support == 0


class TestHandComputedMixedExample:
    """A small, fully hand-verified example mixing correct and incorrect
    predictions across all four categories, to pin down the exact
    precision/recall/F1/confusion-matrix math.
    """

    PAIRS = [
        (NOTIFY, NOTIFY),
        (NOTIFY, NOTIFY),
        (NOTIFY, DIGEST),
        (DIGEST, DIGEST),
        (DIGEST, MUTE),
        (MUTE, MUTE),
        (QUARANTINE, QUARANTINE),
    ]

    def test_confusion_matrix(self):
        matrix = ConfusionMatrix.from_pairs(self.PAIRS)

        assert matrix.matrix[NOTIFY][NOTIFY] == 2
        assert matrix.matrix[NOTIFY][DIGEST] == 1
        assert matrix.matrix[DIGEST][DIGEST] == 1
        assert matrix.matrix[DIGEST][MUTE] == 1
        assert matrix.matrix[MUTE][MUTE] == 1
        assert matrix.matrix[QUARANTINE][QUARANTINE] == 1

    def test_overall_accuracy(self):
        metrics = compute_classification_metrics(self.PAIRS)

        assert metrics.total == 7
        assert metrics.correct == 5
        assert metrics.accuracy == pytest.approx(5 / 7)

    def test_notify_metrics(self):
        metrics = compute_classification_metrics(self.PAIRS)
        notify = metrics.per_category[NOTIFY]

        assert notify.support == 3
        assert notify.precision == pytest.approx(1.0)
        assert notify.recall == pytest.approx(2 / 3)
        assert notify.f1 == pytest.approx(0.8)

    def test_digest_metrics(self):
        metrics = compute_classification_metrics(self.PAIRS)
        digest = metrics.per_category[DIGEST]

        assert digest.support == 2
        assert digest.precision == pytest.approx(0.5)
        assert digest.recall == pytest.approx(0.5)
        assert digest.f1 == pytest.approx(0.5)

    def test_mute_metrics(self):
        metrics = compute_classification_metrics(self.PAIRS)
        mute = metrics.per_category[MUTE]

        assert mute.support == 1
        assert mute.precision == pytest.approx(0.5)
        assert mute.recall == pytest.approx(1.0)
        assert mute.f1 == pytest.approx(2 / 3)

    def test_quarantine_metrics(self):
        metrics = compute_classification_metrics(self.PAIRS)
        quarantine = metrics.per_category[QUARANTINE]

        assert quarantine.support == 1
        assert quarantine.precision == pytest.approx(1.0)
        assert quarantine.recall == pytest.approx(1.0)
        assert quarantine.f1 == pytest.approx(1.0)

    def test_macro_averages(self):
        metrics = compute_classification_metrics(self.PAIRS)

        assert metrics.macro_precision == pytest.approx((1.0 + 0.5 + 0.5 + 1.0) / 4)
        assert metrics.macro_recall == pytest.approx((2 / 3 + 0.5 + 1.0 + 1.0) / 4)
        assert metrics.macro_f1 == pytest.approx((0.8 + 0.5 + 2 / 3 + 1.0) / 4)


class TestClassificationVsDecisionMetricsAreDistinctTypes:
    def test_compute_classification_and_decision_return_different_types(self):
        pairs = [(NOTIFY, NOTIFY)]

        classification_metrics = compute_classification_metrics(pairs)
        decision_metrics = compute_decision_metrics(pairs)

        assert type(classification_metrics).__name__ == "ClassificationMetrics"
        assert type(decision_metrics).__name__ == "DecisionMetrics"
        assert type(classification_metrics) is not type(decision_metrics)

    def test_same_pairs_give_the_same_numeric_results_in_both(self):
        pairs = [(NOTIFY, NOTIFY), (DIGEST, MUTE)]

        classification_metrics = compute_classification_metrics(pairs)
        decision_metrics = compute_decision_metrics(pairs)

        assert classification_metrics.accuracy == decision_metrics.accuracy
        assert classification_metrics.confusion_matrix == decision_metrics.confusion_matrix
