"""Phase 2 dataset loading and validation tests.

Tests read the primary CSV only. They do not modify any dataset files.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    load_fraud_dataset,
    validate_fraud_dataset,
)
from src.utils.paths import RAW_DATASET_PATH


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_fraud_dataset()


def test_dataset_exists() -> None:
    assert RAW_DATASET_PATH.exists(), (
        "Expected data/raw/financial_fraud_detection_dataset.csv relative to "
        "the project root."
    )
    assert RAW_DATASET_PATH.is_file()


def test_required_columns_exist(dataset: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_target_exists(dataset: pd.DataFrame) -> None:
    assert TARGET_COLUMN in dataset.columns


def test_target_is_binary(dataset: pd.DataFrame) -> None:
    values = set(pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce").dropna().unique())
    assert values.issubset({0, 1}), f"Target values are not binary: {values}"
    assert values == {0, 1} or values.issubset({0, 1})
    report = validate_fraud_dataset(dataset)
    assert report.profile["target_is_binary"] is True


def test_dataset_contains_rows(dataset: pd.DataFrame) -> None:
    assert len(dataset) > 0


def test_transaction_ids_are_unique(dataset: pd.DataFrame) -> None:
    assert dataset["Transaction_ID"].is_unique
    assert int(dataset["Transaction_ID"].duplicated().sum()) == 0


def test_fraud_count_greater_than_zero(dataset: pd.DataFrame) -> None:
    fraud_count = int((pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce") == 1).sum())
    assert fraud_count > 0


def test_fraud_rate_between_zero_and_one(dataset: pd.DataFrame) -> None:
    target = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    fraud_rate = float((target == 1).mean())
    assert 0 < fraud_rate < 1


def test_load_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_fraud_dataset(missing)


def test_missing_required_column_fails_validation(dataset: pd.DataFrame) -> None:
    reduced = dataset.drop(columns=[TARGET_COLUMN])
    report = validate_fraud_dataset(reduced)
    assert report.passed is False
    assert any("Missing required columns" in item for item in report.errors)
