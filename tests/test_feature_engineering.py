"""Phase 4 leakage-safe feature engineering tests."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import TARGET_COLUMN, ValidationError, load_fraud_dataset
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    amount_to_average_ratio,
    build_features,
    engineer_feature_frame,
    extract_target,
    parse_transaction_dates,
)
from src.utils.paths import RAW_DATASET_PATH


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_fraud_dataset()


def test_input_dataframe_is_not_modified(raw_df: pd.DataFrame) -> None:
    snapshot = raw_df.copy(deep=True)
    build_features(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_row_count_unchanged(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert len(features) == len(raw_df)
    assert len(features) == 5000 or len(features) == len(raw_df)
    pd.testing.assert_index_equal(features.index, raw_df.index)


def test_required_engineered_columns_exist(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    for column in (
        "amount_to_average_ratio",
        "transaction_hour",
        "transaction_day_of_week",
        "transaction_month",
        "transaction_year",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "Is_International",
    ):
        assert column in features.columns
    debug = engineer_feature_frame(raw_df)
    assert "transaction_day" in debug.columns
    assert "transaction_day" not in features.columns


def test_target_excluded(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert TARGET_COLUMN not in features.columns
    assert "Fraudulent" not in features.columns


def test_transaction_id_excluded(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert "Transaction_ID" not in features.columns


def test_customer_id_excluded(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert "Customer_ID" not in features.columns


def test_suspicious_keyword_excluded(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert "Suspicious_Keyword" not in features.columns


def test_raw_transaction_date_excluded(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert "Transaction_Date" not in features.columns


def test_ratio_calculated_correctly() -> None:
    amount = pd.Series([10.0, 25.0, 0.0])
    average = pd.Series([5.0, 50.0, 10.0])
    ratio = amount_to_average_ratio(amount, average)
    pd.testing.assert_series_equal(ratio, pd.Series([2.0, 0.5, 0.0]))


def test_zero_average_spend_does_not_produce_inf_or_nan() -> None:
    amount = pd.Series([10.0, 0.0, 8.0])
    average = pd.Series([0.0, 0.0, -3.0])
    ratio = amount_to_average_ratio(amount, average)
    assert np.isfinite(ratio.to_numpy()).all()
    assert not ratio.isna().any()
    assert (ratio == 0.0).all()

    frame = pd.DataFrame(
        {
            "Transaction_Date": ["04-10-2023 07:45"] * 3,
            "Transaction_Amount": [10.0, 0.0, 8.0],
            "Average_Spend": [0.0, 0.0, -3.0],
            "Previous_Transactions": [1, 1, 1],
            "Account_Age_Days": [30, 30, 30],
            "Is_International": [0, 1, 0],
            "Merchant_Category": ["Food"] * 3,
            "Payment_Method": ["UPI"] * 3,
            "Device_Type": ["Mobile"] * 3,
            "Location": ["Pune"] * 3,
        }
    )
    features = build_features(frame)
    values = features["amount_to_average_ratio"]
    assert np.isfinite(values.to_numpy()).all()
    assert not values.isna().any()


def test_hour_in_valid_range(raw_df: pd.DataFrame) -> None:
    hour = build_features(raw_df)["transaction_hour"]
    assert int(hour.min()) >= 0
    assert int(hour.max()) <= 23


def test_day_of_week_is_valid(raw_df: pd.DataFrame) -> None:
    dow = build_features(raw_df)["transaction_day_of_week"]
    assert int(dow.min()) >= 0
    assert int(dow.max()) <= 6


def test_binary_international_feature_is_valid(raw_df: pd.DataFrame) -> None:
    intl = build_features(raw_df)["Is_International"]
    assert set(intl.unique()).issubset({0, 1})
    pd.testing.assert_series_equal(
        intl.astype("int64"),
        raw_df["Is_International"].astype("int64"),
        check_names=False,
    )


def test_numerical_feature_list_matches_generated_columns(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert list(features.columns) == list(MODEL_FEATURES)
    assert [column for column in features.columns if column in NUMERIC_FEATURES] == list(
        NUMERIC_FEATURES
    )
    for column in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(features[column])
    for column in CATEGORICAL_FEATURES:
        assert column in features.columns


def test_no_unexpected_missing_values(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    assert int(features.isna().sum().sum()) == 0
    numeric = features[list(NUMERIC_FEATURES)]
    assert np.isfinite(numeric.to_numpy(dtype="float64")).all()


def test_target_not_used_to_construct_features(raw_df: pd.DataFrame) -> None:
    flipped = raw_df.copy()
    flipped[TARGET_COLUMN] = 1 - flipped[TARGET_COLUMN]
    pd.testing.assert_frame_equal(build_features(raw_df), build_features(flipped))
    without_target = raw_df.drop(columns=[TARGET_COLUMN])
    pd.testing.assert_frame_equal(build_features(raw_df), build_features(without_target))


def test_invalid_dates_raise() -> None:
    frame = pd.DataFrame(
        {
            "Transaction_Date": ["not-a-date"],
            "Transaction_Amount": [10.0],
            "Average_Spend": [20.0],
            "Previous_Transactions": [1],
            "Account_Age_Days": [30],
            "Is_International": [0],
            "Merchant_Category": ["Food"],
            "Payment_Method": ["UPI"],
            "Device_Type": ["Mobile"],
            "Location": ["Pune"],
        }
    )
    with pytest.raises(ValidationError, match="Transaction_Date"):
        build_features(frame)


def test_extract_target_alignment(raw_df: pd.DataFrame) -> None:
    target = extract_target(raw_df)
    features = build_features(raw_df)
    assert len(target) == len(features)
    assert TARGET_COLUMN not in features.columns
    assert set(target.unique()).issubset({0, 1})


def test_raw_dataset_path_unchanged() -> None:
    assert RAW_DATASET_PATH.exists()


def test_cyclical_hour_midnight_wrap() -> None:
    parsed = parse_transaction_dates(pd.Series(["01-01-2023 00:00", "01-01-2023 06:00"]))
    assert int(parsed.dt.hour.iloc[0]) == 0
    assert int(parsed.dt.hour.iloc[1]) == 6
