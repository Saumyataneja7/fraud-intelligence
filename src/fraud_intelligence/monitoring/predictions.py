from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_intelligence.monitoring.drift import (
    DriftMonitoringError,
    calculate_psi,
)


@dataclass(frozen=True)
class PredictionDriftResult:
    psi: float
    status: str
    reference_positive_rate: float
    current_positive_rate: float
    reference_count: int
    current_count: int
    classification_threshold: float


def _validate_probabilities(
    values: pd.Series | np.ndarray,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values)

    if probabilities.ndim != 1:
        raise DriftMonitoringError(f"{name} probabilities must be one-dimensional.")

    if probabilities.size == 0:
        raise DriftMonitoringError(f"{name} probabilities cannot be empty.")

    try:
        probabilities = probabilities.astype(float)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(f"{name} probabilities must be numeric.") from exc

    if not np.isfinite(probabilities).all():
        raise DriftMonitoringError(f"{name} probabilities must contain only finite values.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise DriftMonitoringError(f"{name} probabilities must be between 0 and 1.")

    return probabilities


def _validate_drift_thresholds(
    warning_threshold: float,
    alert_threshold: float,
) -> tuple[float, float]:
    try:
        warning = float(warning_threshold)
        alert = float(alert_threshold)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(
            "Thresholds must be numeric, finite, non-negative, and "
            "alert_threshold must exceed warning_threshold."
        ) from exc

    if (
        isinstance(warning_threshold, (bool, np.bool_))
        or isinstance(alert_threshold, (bool, np.bool_))
        or not np.isfinite(warning)
        or not np.isfinite(alert)
        or warning < 0
        or alert <= warning
    ):
        raise DriftMonitoringError(
            "Thresholds must be numeric, finite, non-negative, and "
            "alert_threshold must exceed warning_threshold."
        )

    return warning, alert


def monitor_prediction_drift(
    reference_probabilities: pd.Series | np.ndarray,
    current_probabilities: pd.Series | np.ndarray,
    *,
    classification_threshold: float,
    bins: int = 10,
    warning_threshold: float = 0.1,
    alert_threshold: float = 0.25,
) -> PredictionDriftResult:
    """Monitor prediction-probability drift and positive prediction rates."""

    if (
        isinstance(classification_threshold, (bool, np.bool_))
        or not isinstance(classification_threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(classification_threshold)
        or not 0.0 <= classification_threshold <= 1.0
    ):
        raise DriftMonitoringError(
            "classification_threshold must be numeric, finite, and between 0 and 1."
        )

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise DriftMonitoringError("bins must be an integer of at least 2.")

    warning_threshold, alert_threshold = _validate_drift_thresholds(
        warning_threshold,
        alert_threshold,
    )

    reference = _validate_probabilities(
        reference_probabilities,
        "Reference",
    )
    current = _validate_probabilities(
        current_probabilities,
        "Current",
    )

    psi = calculate_psi(reference, current, bins=bins)

    if psi >= alert_threshold:
        status = "alert"
    elif psi >= warning_threshold:
        status = "warning"
    else:
        status = "stable"

    return PredictionDriftResult(
        psi=psi,
        status=status,
        reference_positive_rate=float(np.mean(reference >= classification_threshold)),
        current_positive_rate=float(np.mean(current >= classification_threshold)),
        reference_count=int(reference.size),
        current_count=int(current.size),
        classification_threshold=float(classification_threshold),
    )
