"""Leakage-safe feature engineering for the fraud detection project.

This module is the single source of truth for transforming raw transaction
rows into model-ready columns. The same functions must be used by the
future training pipeline and the Streamlit prediction page.

It does not train models, fit scalers/encoders, apply SMOTE, drop rows,
or modify the raw CSV. The target is never used to construct features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DATE_FORMAT, TARGET_COLUMN, ValidationError
from src.utils.paths import ARTIFACTS_DIR, REPORTS_DIR

TWO_PI = 2.0 * np.pi

FEATURE_INPUT_COLUMNS: tuple[str, ...] = (
    "Transaction_Date",
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
    "Is_International",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
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
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
)

MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_FEATURES: tuple[str, ...] = (
    "Transaction_ID",
    "Customer_ID",
    "Suspicious_Keyword",
    "Fraudulent",
    "Transaction_Date",
)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "amount_to_average_ratio",
    "transaction_hour",
    "transaction_day",
    "transaction_day_of_week",
    "transaction_month",
    "transaction_year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

# Day-of-month is derived for completeness/debug but omitted from MODEL_FEATURES.
# EDA did not show a day-of-month association, and month + day-of-week already
# represent calendar structure without a noisy 1-31 field.
DEBUG_ONLY_FEATURES: tuple[str, ...] = ("transaction_day",)


def _require_input_columns(df: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValidationError(
            "Feature engineering is missing required input columns: "
            + ", ".join(missing)
        )


def parse_transaction_dates(series: pd.Series) -> pd.Series:
    """Parse Transaction_Date strictly. Invalid values raise, they are not coerced."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="raise")
    else:
        parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
        invalid = int(parsed.isna().sum())
        if invalid:
            examples = series[parsed.isna()].head(5).astype(str).tolist()
            raise ValidationError(
                f"{invalid} Transaction_Date value(s) could not be parsed with "
                f"format {DATE_FORMAT!r}. Examples: {examples}"
            )
    if parsed.isna().any():
        raise ValidationError("Transaction_Date contains missing timestamps.")
    return parsed


def amount_to_average_ratio(
    amount: pd.Series,
    average_spend: pd.Series,
) -> pd.Series:
    """Transaction_Amount / Average_Spend with a defined finite value if spend <= 0.

    Zero or negative average spend is mapped to 0.0 so the result is never inf,
    -inf, or NaN solely because of a zero denominator. Inputs are not mutated.
    """
    amt = pd.to_numeric(amount, errors="raise").astype("float64")
    avg = pd.to_numeric(average_spend, errors="raise").astype("float64")
    amt_np = amt.to_numpy()
    avg_np = avg.to_numpy()
    ratio_np = np.zeros(len(amt_np), dtype="float64")
    valid = avg_np > 0
    ratio_np[valid] = amt_np[valid] / avg_np[valid]
    ratio = pd.Series(ratio_np, index=amount.index)
    if not np.isfinite(ratio.to_numpy()).all():
        raise ValidationError(
            "amount_to_average_ratio produced non-finite values. "
            "Check Transaction_Amount and Average_Spend."
        )
    return ratio


def _cyclical_encoding(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = TWO_PI * values.astype("float64") / float(period)
    return np.sin(radians), np.cos(radians)


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Return the target series without using it to build features."""
    if TARGET_COLUMN not in df.columns:
        raise ValidationError(f"Target column {TARGET_COLUMN!r} is not present.")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    return target.astype("int64").rename(TARGET_COLUMN)


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Identifier columns for display only. Not used as model features."""
    columns = [column for column in ("Transaction_ID",) if column in df.columns]
    return df.loc[:, columns].copy()


def engineer_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered columns plus model categoricals, excluding leakage fields.

    Includes ``transaction_day`` for debugging. Use ``build_features`` for the
    production model matrix.
    """
    _require_input_columns(df)
    parsed = parse_transaction_dates(df["Transaction_Date"])
    hour = parsed.dt.hour.astype("int16")
    day = parsed.dt.day.astype("int16")
    dow = parsed.dt.dayofweek.astype("int16")
    month = parsed.dt.month.astype("int16")
    year = parsed.dt.year.astype("int16")
    hour_sin, hour_cos = _cyclical_encoding(hour, 24)
    dow_sin, dow_cos = _cyclical_encoding(dow, 7)
    international = pd.to_numeric(df["Is_International"], errors="raise").astype("int8")
    if not set(international.unique()).issubset({0, 1}):
        raise ValidationError(
            "Is_International must be binary 0/1 after feature engineering. "
            f"Unique values: {sorted(international.unique().tolist())}"
        )

    out = pd.DataFrame(
        {
            "Transaction_Amount": pd.to_numeric(df["Transaction_Amount"], errors="raise"),
            "Average_Spend": pd.to_numeric(df["Average_Spend"], errors="raise"),
            "Previous_Transactions": pd.to_numeric(df["Previous_Transactions"], errors="raise"),
            "Account_Age_Days": pd.to_numeric(df["Account_Age_Days"], errors="raise"),
            "amount_to_average_ratio": amount_to_average_ratio(
                df["Transaction_Amount"],
                df["Average_Spend"],
            ),
            "transaction_hour": hour,
            "transaction_day": day,
            "transaction_day_of_week": dow,
            "transaction_month": month,
            "transaction_year": year,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "Is_International": international,
            "Merchant_Category": df["Merchant_Category"].astype("string"),
            "Payment_Method": df["Payment_Method"].astype("string"),
            "Device_Type": df["Device_Type"].astype("string"),
            "Location": df["Location"].astype("string"),
        },
        index=df.index,
    )
    return out


def build_features(df: pd.DataFrame, *, include_metadata: bool = False) -> pd.DataFrame:
    """Build the production feature matrix X.

    Parameters
    ----------
    df:
        Raw transaction rows. The input is never mutated.
    include_metadata:
        If True and Transaction_ID exists, prepend it for dashboard display.
        It is still excluded from MODEL_FEATURES.

    Returns
    -------
    DataFrame aligned to ``df.index`` with MODEL_FEATURES (and optional ID).
    """
    frame = engineer_feature_frame(df)
    features = frame.loc[:, list(MODEL_FEATURES)].copy()
    if include_metadata and "Transaction_ID" in df.columns:
        features.insert(0, "Transaction_ID", df["Transaction_ID"].to_numpy())
    validate_feature_matrix(features, expected_rows=len(df), allow_metadata=include_metadata)
    return features


def engineer_features(df: pd.DataFrame, *, include_metadata: bool = False) -> pd.DataFrame:
    """Alias for ``build_features`` so training and serving share one entry point."""
    return build_features(df, include_metadata=include_metadata)


def validate_feature_matrix(
    features: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    allow_metadata: bool = False,
) -> None:
    """Raise ``ValidationError`` if the feature matrix is unsafe for modeling."""
    errors: list[str] = []
    columns = list(features.columns)
    forbidden = {
        TARGET_COLUMN,
        "Customer_ID",
        "Suspicious_Keyword",
        "Transaction_Date",
    }
    if not allow_metadata:
        forbidden.add("Transaction_ID")
    present_forbidden = [column for column in columns if column in forbidden]
    if present_forbidden:
        errors.append("Forbidden columns present in X: " + ", ".join(present_forbidden))

    missing_model = [column for column in MODEL_FEATURES if column not in columns]
    if missing_model:
        errors.append("Missing required model features: " + ", ".join(missing_model))

    unexpected = [
        column
        for column in columns
        if column not in MODEL_FEATURES and not (allow_metadata and column == "Transaction_ID")
    ]
    if unexpected:
        errors.append("Unexpected columns in X: " + ", ".join(unexpected))

    if expected_rows is not None and len(features) != expected_rows:
        errors.append(
            f"Feature row count changed: expected {expected_rows}, got {len(features)}."
        )

    model_frame = features.loc[:, list(MODEL_FEATURES)]
    nan_counts = model_frame.isna().sum()
    nan_total = int(nan_counts.sum())
    if nan_total:
        details = {col: int(count) for col, count in nan_counts.items() if count}
        errors.append(f"Unexpected missing values in X: {details}")

    for column in NUMERIC_FEATURES:
        if column in model_frame.columns and not pd.api.types.is_numeric_dtype(model_frame[column]):
            errors.append(f"Numeric feature {column!r} is not numeric.")

    for column in CATEGORICAL_FEATURES:
        if column not in model_frame.columns:
            errors.append(f"Categorical feature {column!r} is missing.")

    if "transaction_hour" in model_frame.columns:
        hour = model_frame["transaction_hour"]
        if int(hour.min()) < 0 or int(hour.max()) > 23:
            errors.append("transaction_hour is outside 0-23.")

    if "transaction_day_of_week" in model_frame.columns:
        dow = model_frame["transaction_day_of_week"]
        if int(dow.min()) < 0 or int(dow.max()) > 6:
            errors.append("transaction_day_of_week is outside 0-6 (Monday=0).")

    if "Is_International" in model_frame.columns:
        values = set(pd.to_numeric(model_frame["Is_International"], errors="coerce").unique())
        if not values.issubset({0, 1}):
            errors.append(f"Is_International is not binary 0/1: {values}")

    numeric_block = model_frame[list(NUMERIC_FEATURES)]
    if not np.isfinite(numeric_block.to_numpy(dtype="float64")).all():
        errors.append("Numeric features contain inf or -inf.")

    if errors:
        raise ValidationError(
            "Feature matrix validation failed:\n" + "\n".join(f"  - {item}" for item in errors)
        )


def write_feature_preview(
    df: pd.DataFrame,
    path: Path | None = None,
    n_rows: int = 20,
) -> Path:
    """Write a small engineered sample for debugging. Does not write the raw CSV."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(path) if path is not None else ARTIFACTS_DIR / "feature_preview.csv"
    preview = build_features(df.head(n_rows), include_metadata=True)
    preview.to_csv(target, index=False)
    return target


def write_feature_engineering_summary(
    df: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Write reports/feature_engineering_summary.md from the actual dataset."""
    features = build_features(df)
    engineered = engineer_feature_frame(df)
    ratio = features["amount_to_average_ratio"]
    parsed = parse_transaction_dates(df["Transaction_Date"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(path) if path is not None else REPORTS_DIR / "feature_engineering_summary.md"
    lines = [
        "# Phase 4 Feature Engineering Summary",
        "",
        "No models were trained in this phase. No SMOTE, scaling, or encoding was applied.",
        "The raw CSV is not modified. Encoders and scalers belong in Phase 5 and must be fit on training data only.",
        "",
        "## Raw columns",
        "",
        ", ".join(f"`{column}`" for column in df.columns),
        "",
        f"- Rows in: {len(df)}",
        f"- Rows out: {len(features)}",
        f"- Parsed date min: {parsed.min()}",
        f"- Parsed date max: {parsed.max()}",
        "",
        "## Engineered columns",
        "",
        ", ".join(f"`{column}`" for column in ENGINEERED_FEATURES),
        "",
        "`transaction_day` is derived but **not** included in `MODEL_FEATURES` (day-of-month was not an EDA association; month and day-of-week already cover calendar structure).",
        "",
        "## Included model features",
        "",
        "### Numeric",
        "",
        "\n".join(f"- `{column}`" for column in NUMERIC_FEATURES),
        "",
        "### Categorical (not one-hot encoded here)",
        "",
        "\n".join(f"- `{column}`" for column in CATEGORICAL_FEATURES),
        "",
        f"Production `build_features()` returns {len(MODEL_FEATURES)} columns: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical.",
        "",
        "## Excluded features",
        "",
        "\n".join(f"- `{column}`" for column in EXCLUDED_FEATURES),
        "",
        "## Reasons for exclusions",
        "",
        "- `Transaction_ID`: unique row identifier. Useful as dashboard metadata only.",
        "- `Customer_ID`:  high cardinality; median of one transaction per customer. Raw IDs do not generalise. Naive full-data aggregates (`customer_total_transactions`, `customer_average_amount`, `customer_fraud_rate`) would leak future or target information.",
        "- `Suspicious_Keyword`: strong observed association with fraud, but generation time is undocumented. Held out of production X pending provenance review.",
        "- `Fraudulent`: target. Never included in X.",
        "- `Transaction_Date`: raw timestamp string is replaced by hour/dow/month/year and cyclical encodings.",
        "",
        "## Leakage protections",
        "",
        "- Features are row-local. No group-by on the full file, no target encoding, no fraud rates, no customer history from all rows.",
        "- The target is not read by `build_features` / `engineer_feature_frame`.",
        "- No scaler or encoder is fitted. Categorical values stay as labels for a future train-only `ColumnTransformer`.",
        "- Input DataFrame is copied by construction; callers receive a new frame.",
        "- Invalid dates raise instead of becoming silent missing values.",
        "- `validate_feature_matrix` rejects target, IDs, keyword, raw date, inf, and unexpected NaNs.",
        "",
        "## Customer_ID decision",
        "",
        "EXCLUDE raw `Customer_ID` from the production feature matrix.",
        "",
        "Customer-level historical aggregates are a **future enhancement** and must be built from training-time history only, never from the full dataset or from the label.",
        "",
        "## Suspicious_Keyword decision",
        "",
        "EXCLUDE from production ML features. Keep the column in the raw file for EDA/dashboard analysis and provenance investigation. Do not delete it from `data/raw/`.",
        "",
        "## Date/time transformations",
        "",
        "Parsed with format `%d-%m-%Y %H:%M`. Derived: `transaction_hour` (0-23), `transaction_day` (debug only), `transaction_day_of_week` (Monday=0), `transaction_month`, `transaction_year`.",
        "",
        "Cyclical encodings: `hour_sin`/`hour_cos` = sin/cos(2π hour / 24); `dow_sin`/`dow_cos` = sin/cos(2π dow / 7). These let linear models treat 23:00 as close to 00:00.",
        "",
        "Month and year are kept as numeric fields. They do **not** represent proven seasonality; the observed window is limited and the final month is partial.",
        "",
        "## Amount / average ratio",
        "",
        f"- Mean: {ratio.mean():.4f}",
        f"- Median: {ratio.median():.4f}",
        f"- Min: {ratio.min():.4f}",
        f"- Max: {ratio.max():.4f}",
        f"- 99th percentile: {ratio.quantile(0.99):.4f}",
        f"- Non-finite values: {int((~np.isfinite(ratio.to_numpy())).sum())}",
        f"- Rows with Average_Spend <= 0 in this file: {int((pd.to_numeric(df['Average_Spend']) <= 0).sum())}",
        "",
        "If `Average_Spend <= 0`, the ratio is defined as `0.0` so serving code never emits inf/NaN. Extreme ratios are retained (not dropped). No `log1p` transform is applied yet: the ratio is interpretable, and scaling/log choices belong in Phase 5 if training evidence supports them.",
        "",
        "No rule of the form `amount > 200000 = fraud` is implemented. The amount feature is the raw value plus the ratio only.",
        "",
        "## Future feature ideas",
        "",
        "- Train-only customer history (prior count, prior mean amount) using timestamps strictly before the current row.",
        "- `log1p(Transaction_Amount)` only if Phase 5 diagnostics show it helps a linear model.",
        "- Revisit `Suspicious_Keyword` if documentation proves it is known before the decision.",
        "- `transaction_day` cyclical encoding if a later EDA slice shows a month-day pattern.",
        "",
        "ML training and dashboard implementation are planned for later phases.",
        "",
    ]
    _ = engineered  # used to confirm the debug frame can be built from the same rows
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return target
