from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_intelligence.features.dataset import get_model_feature_columns


class DriftMonitoringError(ValueError):
    """Raised when drift-monitoring inputs are invalid."""


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    psi: float
    status: str
    reference_missing_rate: float
    current_missing_rate: float
    reference_count: int
    current_count: int


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI using quantile bins derived from reference values."""

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise DriftMonitoringError("bins must be an integer of at least 2.")

    if isinstance(epsilon, (bool, np.bool_)):
        raise DriftMonitoringError("epsilon must be finite and positive.")

    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError("epsilon must be finite and positive.") from exc

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise DriftMonitoringError("epsilon must be finite and positive.")

    if len(reference) == 0 or len(current) == 0:
        raise DriftMonitoringError("Reference and current data cannot be empty.")

    try:
        reference_values = pd.to_numeric(
            pd.Series(reference),
            errors="raise",
        ).to_numpy(dtype=float)
        current_values = pd.to_numeric(
            pd.Series(current),
            errors="raise",
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(
            "Reference and current data must contain numeric values."
        ) from exc

    if not np.isfinite(reference_values).all():
        raise DriftMonitoringError("Reference data contains non-finite values.")

    if not np.isfinite(current_values).all():
        raise DriftMonitoringError("Current data contains non-finite values.")

    if np.all(reference_values == reference_values[0]):
        reference_proportions = np.array([1.0])
        current_proportions = np.array([np.mean(current_values == reference_values[0])])
        current_proportions = np.append(
            current_proportions,
            1.0 - current_proportions[0],
        )
        reference_proportions = np.append(
            reference_proportions,
            0.0,
        )
    else:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(reference_values, quantiles))

        if edges.size < 2:
            raise DriftMonitoringError("Unable to derive bins from reference data.")

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_counts, _ = np.histogram(reference_values, bins=edges)
        current_counts, _ = np.histogram(current_values, bins=edges)

        reference_proportions = reference_counts / reference_values.size
        current_proportions = current_counts / current_values.size

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )
    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    reference_proportions /= reference_proportions.sum()
    current_proportions /= current_proportions.sum()

    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )

    return float(psi)


def monitor_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    *,
    bins: int = 10,
    warning_threshold: float = 0.1,
    alert_threshold: float = 0.25,
) -> list[FeatureDriftResult]:
    """Compare selected numeric feature distributions across two datasets."""

    if reference.empty or current.empty:
        raise DriftMonitoringError("Reference and current datasets cannot be empty.")

    if not features:
        raise DriftMonitoringError("At least one feature must be selected.")

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise DriftMonitoringError("bins must be an integer of at least 2.")

    if len(set(features)) != len(features):
        raise DriftMonitoringError("Duplicate features are not allowed.")

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

    warning_threshold = warning
    alert_threshold = alert

    missing_columns = set(features) - (set(reference.columns) & set(current.columns))
    if missing_columns:
        raise DriftMonitoringError(f"Requested features are missing: {sorted(missing_columns)}")

    approved_features = set(
        get_model_feature_columns(set(reference.columns) & set(current.columns))
    )
    unapproved_features = set(features) - approved_features
    if unapproved_features:
        raise DriftMonitoringError(
            f"Only approved model features can be monitored: {sorted(unapproved_features)}"
        )

    results: list[FeatureDriftResult] = []

    for feature in features:
        reference_series = reference[feature]
        current_series = current[feature]

        if not pd.api.types.is_numeric_dtype(reference_series):
            raise DriftMonitoringError(f"Reference feature '{feature}' must be numeric.")
        if not pd.api.types.is_numeric_dtype(current_series):
            raise DriftMonitoringError(f"Current feature '{feature}' must be numeric.")

        reference_numeric = reference_series.to_numpy(dtype=float, na_value=np.nan)
        current_numeric = current_series.to_numpy(dtype=float, na_value=np.nan)

        if not np.isfinite(reference_numeric[~np.isnan(reference_numeric)]).all():
            raise DriftMonitoringError(f"Reference feature '{feature}' contains non-finite values.")
        if not np.isfinite(current_numeric[~np.isnan(current_numeric)]).all():
            raise DriftMonitoringError(f"Current feature '{feature}' contains non-finite values.")

        reference_missing_rate = float(np.isnan(reference_numeric).mean())
        current_missing_rate = float(np.isnan(current_numeric).mean())

        reference_values = reference_numeric[~np.isnan(reference_numeric)]
        current_values = current_numeric[~np.isnan(current_numeric)]

        if reference_values.size == 0 or current_values.size == 0:
            raise DriftMonitoringError(
                f"Feature '{feature}' must have non-missing values in both datasets."
            )

        psi = calculate_psi(
            reference_values,
            current_values,
            bins=bins,
        )

        if psi >= alert_threshold:
            status = "alert"
        elif psi >= warning_threshold:
            status = "warning"
        else:
            status = "stable"

        results.append(
            FeatureDriftResult(
                feature=feature,
                psi=psi,
                status=status,
                reference_missing_rate=reference_missing_rate,
                current_missing_rate=current_missing_rate,
                reference_count=int(len(reference)),
                current_count=int(len(current)),
            )
        )

    return results
