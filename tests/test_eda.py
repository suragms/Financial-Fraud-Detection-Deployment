"""Phase 3 EDA tests. These tests do not modify the raw dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import (
    add_temporary_eda_features,
    raw_dataset_md5,
    safe_amount_to_average_ratio,
)
from src.data.loader import REQUIRED_COLUMNS, TARGET_COLUMN, load_fraud_dataset
from src.utils.paths import RAW_DATASET_PATH


def test_eda_source_dataset_exists() -> None:
    assert RAW_DATASET_PATH.exists()
    assert RAW_DATASET_PATH.is_file()


def test_eda_required_columns_exist() -> None:
    frame = load_fraud_dataset()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    assert not missing


def test_eda_target_is_binary() -> None:
    frame = load_fraud_dataset()
    values = set(pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").dropna().unique())
    assert values.issubset({0, 1})
    assert 0 in values and 1 in values


def test_ratio_handles_zero_denominator() -> None:
    amount = pd.Series([10.0, 5.0, 0.0, 8.0])
    average = pd.Series([2.0, 0.0, 4.0, -1.0])
    ratio = safe_amount_to_average_ratio(amount, average)
    assert ratio.iloc[0] == 5.0
    assert pd.isna(ratio.iloc[1])
    assert ratio.iloc[2] == 0.0
    assert pd.isna(ratio.iloc[3])


def test_ratio_does_not_mutate_inputs() -> None:
    amount = pd.Series([10.0, 20.0])
    average = pd.Series([5.0, 0.0])
    amount_copy = amount.copy()
    average_copy = average.copy()
    safe_amount_to_average_ratio(amount, average)
    pd.testing.assert_series_equal(amount, amount_copy)
    pd.testing.assert_series_equal(average, average_copy)


def test_eda_does_not_alter_raw_dataset() -> None:
    before = raw_dataset_md5()
    frame = load_fraud_dataset()
    original_columns = list(frame.columns)
    enriched = add_temporary_eda_features(frame)
    after = raw_dataset_md5()
    assert before == after
    assert list(frame.columns) == original_columns
    assert "amount_to_average_ratio" in enriched.columns
    assert "amount_to_average_ratio" not in frame.columns
    assert RAW_DATASET_PATH.read_bytes()
    np.testing.assert_array_equal(
        frame["Transaction_Amount"].to_numpy(),
        load_fraud_dataset()["Transaction_Amount"].to_numpy(),
    )


def test_temporary_features_use_a_copy() -> None:
    frame = load_fraud_dataset()
    snapshot = frame.copy(deep=True)
    add_temporary_eda_features(frame)
    pd.testing.assert_frame_equal(frame, snapshot)
