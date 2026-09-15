import pandas as pd
import pytest

from fraud_intelligence.features.temporal_features import (
    add_temporal_features,
)


def make_transactions():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-04 23:30:00+00:00",
                    "2025-01-06 05:30:00+00:00",
                    "2025-07-15 14:30:00+00:00",
                ],
                utc=True,
            )
        }
    )


def test_temporal_features_are_created():
    result = add_temporal_features(
        make_transactions()
    )

    expected_columns = {
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
        "is_night",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_temporal_values():
    result = add_temporal_features(
        make_transactions()
    )

    assert result.loc[0, "hour"] == 23
    assert result.loc[0, "day_of_week"] == 5
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "quarter"] == 1
    assert result.loc[0, "is_weekend"] == 1
    assert result.loc[0, "is_night"] == 1


def test_daytime_transaction():
    result = add_temporal_features(
        make_transactions()
    )

    assert result.loc[2, "hour"] == 14
    assert result.loc[2, "is_weekend"] == 0
    assert result.loc[2, "is_night"] == 0


def test_timezone_is_required():
    transactions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 12:00:00"]
            )
        }
    )

    with pytest.raises(TypeError):
        add_temporal_features(
            transactions
        )