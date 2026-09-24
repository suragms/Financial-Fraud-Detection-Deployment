"""Data loading, validation, and EDA helpers."""

from src.data.loader import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    ValidationError,
    ValidationReport,
    format_validation_report,
    load_fraud_dataset,
    validate_fraud_dataset,
)

_EDA_EXPORTS = {
    "raw_dataset_md5",
    "safe_amount_to_average_ratio",
    "add_temporary_eda_features",
    "categorical_fraud_summary",
}

__all__ = [
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "ValidationError",
    "ValidationReport",
    "format_validation_report",
    "load_fraud_dataset",
    "validate_fraud_dataset",
    "raw_dataset_md5",
    "safe_amount_to_average_ratio",
    "add_temporary_eda_features",
    "categorical_fraud_summary",
]


def __getattr__(name: str):
    if name in _EDA_EXPORTS:
        from src.data import eda

        return getattr(eda, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
