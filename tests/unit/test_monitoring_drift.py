import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.monitoring.drift import (
    DriftMonitoringError,
    calculate_psi,
    monitor_feature_drift,
)


def test_calculate_psi_is_zero_for_identical_distributions():
    values = np.arange(100, dtype=float)

    psi = calculate_psi(values, values.copy())

    assert psi == pytest.approx(0.0, abs=1e-12)


def test_calculate_psi_detects_distribution_shift():
    reference = np.linspace(0, 1, 1000)
    current = np.linspace(5, 6, 1000)

    psi = calculate_psi(reference, current)

    assert psi > 0.25


@pytest.mark.parametrize("bins", [0, 1, -1])
def test_calculate_psi_rejects_invalid_bin_count(bins):
    with pytest.raises(DriftMonitoringError, match="bins"):
        calculate_psi([1, 2, 3], [1, 2, 3], bins=bins)


@pytest.mark.parametrize("epsilon", [0, -1, np.inf, np.nan])
def test_calculate_psi_rejects_invalid_epsilon(epsilon):
    with pytest.raises(DriftMonitoringError, match="epsilon"):
        calculate_psi([1, 2, 3], [1, 2, 3], epsilon=epsilon)


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        ([], [1, 2]),
        ([1, 2], []),
        ([np.nan, np.nan], [1, 2]),
        ([1, 2], [np.inf, 3]),
    ],
)
def test_calculate_psi_rejects_empty_or_non_finite_data(reference, current):
    with pytest.raises(DriftMonitoringError):
        calculate_psi(reference, current)


def test_calculate_psi_handles_constant_reference_values():
    reference = np.full(100, 5.0)
    current = np.full(100, 5.0)

    assert calculate_psi(reference, current) == pytest.approx(0.0, abs=1e-12)


def test_monitor_feature_drift_reports_stable_feature_and_missing_rates():
    reference = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, np.nan, 4.0]})
    current = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, np.nan, 4.0]})

    results = monitor_feature_drift(reference, current, ["customer_txn_count_5m"])

    assert len(results) == 1
    assert results[0].feature == "customer_txn_count_5m"
    assert results[0].psi == pytest.approx(0.0, abs=1e-12)
    assert results[0].status == "stable"
    assert results[0].reference_missing_rate == pytest.approx(0.25)
    assert results[0].current_missing_rate == pytest.approx(0.25)
    assert results[0].reference_count == 4
    assert results[0].current_count == 4


def test_monitor_feature_drift_flags_shifted_feature():
    reference = pd.DataFrame({"customer_txn_count_5m": np.linspace(0, 1, 1000)})
    current = pd.DataFrame({"customer_txn_count_5m": np.linspace(5, 6, 1000)})

    results = monitor_feature_drift(reference, current, ["customer_txn_count_5m"])

    assert results[0].status == "alert"
    assert results[0].psi >= 0.25


def test_monitor_feature_drift_rejects_target_column():
    reference = pd.DataFrame({"is_fraud": [0, 1, 0, 1]})
    current = pd.DataFrame({"is_fraud": [0, 0, 1, 1]})

    with pytest.raises(DriftMonitoringError, match="approved model features"):
        monitor_feature_drift(reference, current, ["is_fraud"])


def test_monitor_feature_drift_rejects_identifier_column():
    reference = pd.DataFrame({"transaction_id": [1, 2, 3]})
    current = pd.DataFrame({"transaction_id": [4, 5, 6]})

    with pytest.raises(DriftMonitoringError, match="approved model features"):
        monitor_feature_drift(reference, current, ["transaction_id"])


def test_monitor_feature_drift_rejects_missing_columns():
    reference = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0]})
    current = pd.DataFrame({"other_feature": [1.0, 2.0]})

    with pytest.raises(DriftMonitoringError, match="missing"):
        monitor_feature_drift(reference, current, ["customer_txn_count_5m"])


def test_monitor_feature_drift_rejects_duplicate_features():
    frame = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, 3.0]})

    with pytest.raises(DriftMonitoringError, match="Duplicate"):
        monitor_feature_drift(frame, frame, ["customer_txn_count_5m", "customer_txn_count_5m"])


def test_monitor_feature_drift_rejects_invalid_thresholds():
    frame = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, 3.0]})

    with pytest.raises(DriftMonitoringError, match="Thresholds"):
        monitor_feature_drift(
            frame,
            frame,
            ["customer_txn_count_5m"],
            warning_threshold=0.25,
            alert_threshold=0.1,
        )


def test_monitor_feature_drift_rejects_non_numeric_unapproved_feature():
    frame = pd.DataFrame({"merchant_category": ["grocery", "travel"]})

    with pytest.raises(DriftMonitoringError, match="approved model features"):
        monitor_feature_drift(frame, frame, ["merchant_category"])


@pytest.mark.parametrize("bins", [0, 1, -1])
def test_monitor_feature_drift_rejects_invalid_bin_count(bins):
    frame = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, 3.0]})

    with pytest.raises(DriftMonitoringError, match="bins"):
        monitor_feature_drift(frame, frame, ["customer_txn_count_5m"], bins=bins)


@pytest.mark.parametrize("bins", [2.5, "10", None])
def test_calculate_psi_rejects_non_integer_bin_count(bins):
    with pytest.raises(DriftMonitoringError, match="bins"):
        calculate_psi([1, 2, 3], [1, 2, 3], bins=bins)


def test_calculate_psi_detects_shift_from_constant_reference():
    reference = np.full(100, 5.0)
    current = np.full(100, 10.0)

    assert calculate_psi(reference, current) > 0.25


def test_monitor_feature_drift_rejects_infinite_feature_values():
    reference = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"customer_txn_count_5m": [1.0, np.inf, 3.0]})

    with pytest.raises(DriftMonitoringError, match="non-finite"):
        monitor_feature_drift(reference, current, ["customer_txn_count_5m"])


def test_monitor_feature_drift_rejects_feature_with_only_missing_values():
    reference = pd.DataFrame({"customer_txn_count_5m": [np.nan, np.nan]})
    current = pd.DataFrame({"customer_txn_count_5m": [1.0, 2.0]})

    with pytest.raises(DriftMonitoringError, match="non-missing"):
        monitor_feature_drift(reference, current, ["customer_txn_count_5m"])
