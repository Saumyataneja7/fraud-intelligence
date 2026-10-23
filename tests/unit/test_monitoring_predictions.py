import numpy as np
import pytest

from fraud_intelligence.monitoring.drift import DriftMonitoringError
from fraud_intelligence.monitoring.predictions import monitor_prediction_drift


def test_monitor_prediction_drift_reports_stable_distribution_and_rates():
    probabilities = np.linspace(0.0, 1.0, 100)

    result = monitor_prediction_drift(
        probabilities,
        probabilities.copy(),
        classification_threshold=0.5,
    )

    assert result.psi == pytest.approx(0.0, abs=1e-12)
    assert result.status == "stable"
    assert result.reference_positive_rate == pytest.approx(0.5)
    assert result.current_positive_rate == pytest.approx(0.5)
    assert result.reference_count == 100
    assert result.current_count == 100
    assert result.classification_threshold == 0.5


def test_monitor_prediction_drift_detects_distribution_shift():
    reference = np.linspace(0.0, 0.2, 1000)
    current = np.linspace(0.8, 1.0, 1000)

    result = monitor_prediction_drift(
        reference,
        current,
        classification_threshold=0.5,
    )

    assert result.psi >= 0.25
    assert result.status == "alert"
    assert result.reference_positive_rate == 0.0
    assert result.current_positive_rate == 1.0


def test_monitor_prediction_drift_uses_supplied_classification_threshold():
    reference = np.array([0.2, 0.4, 0.6, 0.8])
    current = np.array([0.1, 0.3, 0.7, 0.9])

    result = monitor_prediction_drift(
        reference,
        current,
        classification_threshold=0.7,
    )

    assert result.classification_threshold == 0.7
    assert result.reference_positive_rate == pytest.approx(0.25)
    assert result.current_positive_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        ([], [0.1, 0.2]),
        ([0.1, 0.2], []),
        ([np.nan, 0.2], [0.1, 0.2]),
        ([0.1, 0.2], [np.inf, 0.3]),
        ([0.1, 0.2], [-np.inf, 0.3]),
        ([-0.1, 0.2], [0.1, 0.2]),
        ([0.1, 1.1], [0.1, 0.2]),
        (["low", "high"], [0.1, 0.2]),
    ],
)
def test_monitor_prediction_drift_rejects_invalid_probabilities(
    reference,
    current,
):
    with pytest.raises(DriftMonitoringError):
        monitor_prediction_drift(
            reference,
            current,
            classification_threshold=0.5,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, np.nan, np.inf, -np.inf, True, "0.5", None],
)
def test_monitor_prediction_drift_rejects_invalid_classification_threshold(
    threshold,
):
    with pytest.raises(DriftMonitoringError):
        monitor_prediction_drift(
            [0.1, 0.2],
            [0.1, 0.2],
            classification_threshold=threshold,
        )


@pytest.mark.parametrize(
    ("warning_threshold", "alert_threshold"),
    [
        (-0.1, 0.25),
        (0.1, 0.1),
        (0.2, 0.1),
        (np.nan, 0.25),
        (0.1, np.inf),
        ("invalid", 0.25),
        (True, 0.25),
    ],
)
def test_monitor_prediction_drift_rejects_invalid_drift_thresholds(
    warning_threshold,
    alert_threshold,
):
    with pytest.raises(DriftMonitoringError, match="Thresholds"):
        monitor_prediction_drift(
            [0.1, 0.2],
            [0.1, 0.2],
            classification_threshold=0.5,
            warning_threshold=warning_threshold,
            alert_threshold=alert_threshold,
        )


@pytest.mark.parametrize("bins", [0, 1, -1, 2.5, "10", None, True])
def test_monitor_prediction_drift_rejects_invalid_bin_count(bins):
    with pytest.raises(DriftMonitoringError, match="bins"):
        monitor_prediction_drift(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            classification_threshold=0.5,
            bins=bins,
        )


def test_monitor_prediction_drift_rejects_multidimensional_probabilities():
    with pytest.raises(DriftMonitoringError, match="one-dimensional"):
        monitor_prediction_drift(
            [[0.1, 0.2], [0.3, 0.4]],
            [0.1, 0.2],
            classification_threshold=0.5,
        )
