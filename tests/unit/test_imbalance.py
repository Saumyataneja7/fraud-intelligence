from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.ml.imbalance import (
    ClassImbalanceError,
    build_class_imbalance_strategy,
    build_class_imbalance_summary,
    calculate_class_distribution,
    calculate_class_weights,
    calculate_scale_pos_weight,
)


def make_target() -> pd.Series:
    """Create a deterministic imbalanced binary target."""

    return pd.Series(
        [0] * 90 + [1] * 10,
        name="is_fraud",
    )


def test_class_distribution():
    target = make_target()

    distribution = calculate_class_distribution(target)

    assert distribution.total_rows == 100
    assert distribution.negative_count == 90
    assert distribution.positive_count == 10

    assert distribution.negative_rate == pytest.approx(0.90)
    assert distribution.positive_rate == pytest.approx(0.10)

    assert distribution.imbalance_ratio == pytest.approx(9.0)


def test_balanced_class_weights():
    target = make_target()

    weights = calculate_class_weights(target)

    # Formula:
    # negative = 100 / (2 * 90)
    # positive = 100 / (2 * 10)
    assert weights[0] == pytest.approx(
        100 / (2 * 90)
    )

    assert weights[1] == pytest.approx(
        100 / (2 * 10)
    )


def test_positive_class_receives_higher_weight():
    target = make_target()

    weights = calculate_class_weights(target)

    assert weights[1] > weights[0]


def test_scale_pos_weight():
    target = make_target()

    scale_pos_weight = calculate_scale_pos_weight(target)

    assert scale_pos_weight == pytest.approx(9.0)


def test_complete_strategy():
    target = make_target()

    strategy = build_class_imbalance_strategy(target)

    assert strategy.class_weights[0] == pytest.approx(
        100 / (2 * 90)
    )

    assert strategy.class_weights[1] == pytest.approx(
        100 / (2 * 10)
    )

    assert strategy.scale_pos_weight == pytest.approx(9.0)

    assert strategy.distribution.total_rows == 100


def test_sklearn_class_weight_is_copy():
    target = make_target()

    strategy = build_class_imbalance_strategy(target)

    weights = strategy.sklearn_class_weight

    assert weights == strategy.class_weights
    assert weights is not strategy.class_weights


def test_xgboost_weight_property():
    target = make_target()

    strategy = build_class_imbalance_strategy(target)

    assert (
        strategy.xgboost_scale_pos_weight
        == pytest.approx(9.0)
    )


def test_summary():
    target = make_target()

    strategy = build_class_imbalance_strategy(target)
    summary = build_class_imbalance_summary(strategy)

    assert summary["strategy"] == (
        "training-set class weighting"
    )

    assert summary["target_column"] == "is_fraud"

    assert summary["total_training_rows"] == 100
    assert summary["negative_count"] == 90
    assert summary["positive_count"] == 10

    assert summary["positive_rate"] == pytest.approx(0.10)

    assert summary["imbalance_ratio"] == pytest.approx(9.0)

    assert summary["validation_rebalanced"] is False
    assert summary["test_rebalanced"] is False


def test_missing_values_raise():
    target = pd.Series(
        [0, 1, None],
        name="is_fraud",
    )

    with pytest.raises(ClassImbalanceError):
        calculate_class_distribution(target)


def test_non_binary_target_raises():
    target = pd.Series(
        [0, 1, 2, 0],
        name="is_fraud",
    )

    with pytest.raises(ClassImbalanceError):
        calculate_class_distribution(target)


def test_empty_target_raises():
    target = pd.Series(
        [],
        dtype="int64",
        name="is_fraud",
    )

    with pytest.raises(ClassImbalanceError):
        calculate_class_distribution(target)


def test_single_class_target_raises():
    target = pd.Series(
        [0, 0, 0, 0],
        name="is_fraud",
    )

    with pytest.raises(ClassImbalanceError):
        build_class_imbalance_strategy(target)


def test_strategy_uses_only_training_target():
    train_target = pd.Series(
        [0] * 90 + [1] * 10,
        name="is_fraud",
    )

    # These are deliberately different and should not affect
    # the strategy because they are not supplied to it.
    validation_target = pd.Series(
        [0] * 50 + [1] * 10,
        name="is_fraud",
    )

    test_target = pd.Series(
        [0] * 10 + [1] * 10,
        name="is_fraud",
    )

    strategy = build_class_imbalance_strategy(
        train_target
    )

    assert strategy.distribution.total_rows == 100
    assert strategy.distribution.positive_count == 10
    assert strategy.scale_pos_weight == pytest.approx(9.0)

    assert len(validation_target) == 60
    assert len(test_target) == 20