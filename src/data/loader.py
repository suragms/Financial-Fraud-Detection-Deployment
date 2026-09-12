"""Load and validate the primary financial fraud dataset.

This module does not train models, encode features, drop rows, or modify
the source CSV. ``Suspicious_Keyword`` is treated as a data column only and
must not be used as an ML feature in later phases without a leakage review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import PROJECT_ROOT, RAW_DATASET_PATH, require_path


def _project_relative(path: Path | str | None) -> str:
    """Return a project-relative POSIX path for reports (no user home prefix)."""
    candidate = Path(path) if path is not None else RAW_DATASET_PATH
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(candidate).as_posix()


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Suspicious_Keyword",
    "Fraudulent",
)

IDENTIFIER_COLUMNS: tuple[str, ...] = ("Transaction_ID", "Customer_ID")
NUMERIC_COLUMNS: tuple[str, ...] = (
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Suspicious_Keyword",
)
BINARY_COLUMNS: tuple[str, ...] = ("Is_International", "Fraudulent")
DATETIME_COLUMNS: tuple[str, ...] = ("Transaction_Date",)
TARGET_COLUMN = "Fraudulent"

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
)

DATE_FORMAT = "%d-%m-%Y %H:%M"
ALLOWED_KEYWORD_VALUES = {"Yes", "No"}
BINARY_VALUES = {0, 1}


class ValidationError(ValueError):
    """Raised when a dataset fails critical validation checks."""


@dataclass
class ValidationReport:
    """Human-readable validation outcome. Does not modify the dataset."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.passed:
            details = "\n".join(f"  - {item}" for item in self.errors)
            raise ValidationError(
                "Dataset validation failed with the following issues:\n"
                f"{details}"
            )


def load_fraud_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the primary fraud CSV and return a DataFrame.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if the file cannot be parsed as a table.
    """
    dataset_path = Path(path) if path is not None else RAW_DATASET_PATH
    existing = require_path(dataset_path, kind="dataset file")

    try:
        frame = pd.read_csv(existing)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {existing.name}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Dataset file could not be parsed as CSV: {existing.name}"
        ) from exc

    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"Dataset file has no columns: {existing.name}")

    return frame


def _parse_transaction_dates(series: pd.Series) -> pd.Series:
    """Parse dates for validation only. Does not alter the original column."""
    parsed = pd.to_datetime(
        series,
        format=DATE_FORMAT,
        errors="coerce",
    )
    if parsed.notna().any():
        return parsed

    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _binary_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isin(list(BINARY_VALUES)) & numeric.notna()


def validate_fraud_dataset(
    df: pd.DataFrame,
    *,
    source_path: Path | str | None = None,
) -> ValidationReport:
    """Validate schema, types, and data quality without changing ``df``."""
    errors: list[str] = []
    warnings: list[str] = []

    actual_columns = list(df.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in REQUIRED_COLUMNS]

    if missing_columns:
        errors.append(
            "Missing required columns: " + ", ".join(missing_columns)
        )
    if unexpected_columns:
        warnings.append(
            "Unexpected columns (not renamed or dropped): "
            + ", ".join(unexpected_columns)
        )

    row_count = int(len(df))
    column_count = int(df.shape[1])

    if row_count == 0:
        errors.append("Dataset contains no rows.")

    missing_total = int(df.isna().sum().sum()) if column_count else 0
    if missing_total:
        missing_by_column = {
            col: int(count)
            for col, count in df.isna().sum().items()
            if count
        }
        warnings.append(
            f"Missing values found: {missing_total} cells "
            f"({missing_by_column}). Rows were not deleted."
        )

    duplicate_rows = int(df.duplicated().sum()) if column_count else 0
    if duplicate_rows:
        warnings.append(
            f"Duplicate rows found: {duplicate_rows}. Rows were not deleted."
        )

    duplicate_ids = 0
    if "Transaction_ID" in df.columns:
        duplicate_ids = int(df["Transaction_ID"].duplicated().sum())
        if duplicate_ids:
            errors.append(
                f"Duplicate Transaction_ID values found: {duplicate_ids}."
            )

    parsed_dates = pd.Series(dtype="datetime64[ns]")
    invalid_dates = 0
    date_min = None
    date_max = None
    if "Transaction_Date" in df.columns:
        parsed_dates = _parse_transaction_dates(df["Transaction_Date"])
        invalid_dates = int(parsed_dates.isna().sum())
        if invalid_dates:
            errors.append(
                f"Invalid Transaction_Date values that could not be parsed: "
                f"{invalid_dates}."
            )
        if parsed_dates.notna().any():
            date_min = parsed_dates.min()
            date_max = parsed_dates.max()

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        if not _is_numeric_series(df[column]):
            coerced = pd.to_numeric(df[column], errors="coerce")
            invalid = int(coerced.isna().sum() - df[column].isna().sum())
            errors.append(
                f"Column '{column}' is not numeric "
                f"({invalid} value(s) could not be coerced)."
            )

    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns or not _is_numeric_series(df[column]):
            continue
        negative_count = int((df[column] < 0).sum())
        if negative_count:
            errors.append(
                f"Column '{column}' has {negative_count} negative value(s)."
            )

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue
        invalid_binary = int((~_binary_mask(df[column])).sum())
        if invalid_binary:
            unique_values = sorted(df[column].dropna().unique().tolist(), key=str)
            errors.append(
                f"Column '{column}' must contain only 0 or 1 "
                f"({invalid_binary} invalid value(s); unique values: {unique_values})."
            )

    if "Suspicious_Keyword" in df.columns:
        keyword_values = set(df["Suspicious_Keyword"].dropna().astype(str).unique())
        unexpected_keywords = sorted(keyword_values - ALLOWED_KEYWORD_VALUES)
        if unexpected_keywords:
            warnings.append(
                "Suspicious_Keyword has unexpected values "
                f"{unexpected_keywords}. Expected {sorted(ALLOWED_KEYWORD_VALUES)}. "
                "This column is not used as an ML feature in this phase."
            )

    fraud_count = 0
    legitimate_count = 0
    fraud_rate = None
    target_is_binary = False
    if TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' does not exist.")
    else:
        numeric_target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        target_is_binary = bool(
            numeric_target.notna().all()
            and set(numeric_target.dropna().unique()).issubset(BINARY_VALUES)
        )
        if not target_is_binary:
            errors.append(
                f"Target column '{TARGET_COLUMN}' is not binary with values {{0, 1}}."
            )
        else:
            fraud_count = int((numeric_target == 1).sum())
            legitimate_count = int((numeric_target == 0).sum())
            if row_count:
                fraud_rate = fraud_count / row_count
            if fraud_count == 0:
                errors.append("Fraud count is zero; the target has no positive class.")

    profile = {
        "dataset_path": _project_relative(source_path),
        "rows": row_count,
        "columns": column_count,
        "missing_values": missing_total,
        "duplicate_rows": duplicate_rows,
        "duplicate_transaction_ids": duplicate_ids,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_rate": fraud_rate,
        "date_minimum": date_min,
        "date_maximum": date_max,
        "invalid_dates": invalid_dates,
        "numerical_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "datetime_columns": list(DATETIME_COLUMNS),
        "binary_columns": list(BINARY_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_is_binary": target_is_binary,
        "column_names": actual_columns,
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
    }

    passed = not errors
    return ValidationReport(
        passed=passed,
        errors=errors,
        warnings=warnings,
        profile=profile,
    )


def format_validation_report(report: ValidationReport) -> str:
    """Render a concise human-readable validation report."""
    profile = report.profile
    fraud_rate = profile.get("fraud_rate")
    fraud_rate_text = (
        f"{fraud_rate * 100:.2f}%" if fraud_rate is not None else "n/a"
    )
    date_min = profile.get("date_minimum")
    date_max = profile.get("date_maximum")

    status = "DATA VALIDATION PASSED" if report.passed else "DATA VALIDATION FAILED"
    lines = [
        status,
        "",
        f"Dataset path: {profile.get('dataset_path')}",
        f"Rows: {profile.get('rows')}",
        f"Columns: {profile.get('columns')}",
        f"Missing values: {profile.get('missing_values')}",
        f"Duplicate rows: {profile.get('duplicate_rows')}",
        f"Duplicate Transaction_IDs: {profile.get('duplicate_transaction_ids')}",
        f"Fraud cases: {profile.get('fraud_count')}",
        f"Legitimate transactions: {profile.get('legitimate_count')}",
        f"Fraud rate: {fraud_rate_text}",
        f"Date minimum: {date_min}",
        f"Date maximum: {date_max}",
        "",
        f"Numerical columns: {', '.join(profile.get('numerical_columns', []))}",
        f"Categorical columns: {', '.join(profile.get('categorical_columns', []))}",
        f"Identifier columns: {', '.join(profile.get('identifier_columns', []))}",
        f"Target column: {profile.get('target_column')}",
    ]

    if report.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {item}" for item in report.warnings)

    if report.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"  - {item}" for item in report.errors)

    return "\n".join(lines)
