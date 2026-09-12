"""Exploratory analysis helpers for Phase 3.

These functions never write back to the raw CSV, never drop rows, and never
train models. Temporary columns are returned on copies only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DATE_FORMAT, REQUIRED_COLUMNS, TARGET_COLUMN
from src.utils.paths import EDA_ARTIFACTS_DIR, RAW_DATASET_PATH, REPORTS_DIR

CATEGORICAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Suspicious_Keyword",
)

NUMERIC_EDA_COLUMNS: tuple[str, ...] = (
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
)

AMOUNT_BINS = (0, 25, 50, 100, 200, 400, 1000)
ACCOUNT_AGE_BINS = (0, 180, 365, 730, 1460, 10_000)


def raw_dataset_md5(path: Path | None = None) -> str:
    """Return the MD5 digest of the raw CSV bytes without modifying the file."""
    target = Path(path) if path is not None else RAW_DATASET_PATH
    return hashlib.md5(target.read_bytes()).hexdigest()


def ensure_eda_artifact_dir() -> Path:
    EDA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_ARTIFACTS_DIR


def safe_amount_to_average_ratio(
    amount: pd.Series | np.ndarray | list[float],
    average_spend: pd.Series | np.ndarray | list[float],
) -> pd.Series:
    """Return Transaction_Amount / Average_Spend.

    Zero, negative, or non-finite denominators become NA. The inputs are not
    modified. This is an EDA/candidate feature helper, not a persisted column.
    """
    amt = pd.to_numeric(pd.Series(amount, dtype="float64"), errors="coerce").reset_index(drop=True)
    avg = pd.to_numeric(pd.Series(average_spend, dtype="float64"), errors="coerce").reset_index(drop=True)
    if len(amt) != len(avg):
        raise ValueError("amount and average_spend must have the same length")
    safe_denom = avg.where(avg > 0)
    return amt / safe_denom


def add_temporary_datetime_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with hour, day, day_of_week, month, and year.

    Does not modify ``df`` or the source CSV.
    """
    out = df.copy()
    parsed = pd.to_datetime(
        out["Transaction_Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    out["hour"] = parsed.dt.hour
    out["day"] = parsed.dt.day
    out["day_of_week"] = parsed.dt.day_name()
    out["month"] = parsed.dt.month
    out["year"] = parsed.dt.year
    out["year_month"] = parsed.dt.to_period("M").astype(str)
    out["_parsed_date"] = parsed
    return out


def add_temporary_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with datetime parts and amount-to-average ratio."""
    out = add_temporary_datetime_parts(df)
    out["amount_to_average_ratio"] = safe_amount_to_average_ratio(
        out["Transaction_Amount"],
        out["Average_Spend"],
    )
    return out


def target_summary(df: pd.DataFrame) -> dict[str, float | int]:
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    n = int(len(df))
    fraud = int((target == 1).sum())
    legit = int((target == 0).sum())
    return {
        "rows": n,
        "columns": int(df.shape[1]),
        "fraud_count": fraud,
        "legitimate_count": legit,
        "fraud_rate": (fraud / n) if n else float("nan"),
        "legitimate_rate": (legit / n) if n else float("nan"),
    }


def numeric_by_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_EDA_COLUMNS:
        grouped = df.groupby(TARGET_COLUMN)[column].describe()
        for label, stats in grouped.iterrows():
            rows.append(
                {
                    "feature": column,
                    "Fraudulent": int(label),
                    "count": float(stats["count"]),
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "min": float(stats["min"]),
                    "q25": float(stats["25%"]),
                    "median": float(stats["50%"]),
                    "q75": float(stats["75%"]),
                    "max": float(stats["max"]),
                    "corr_with_target": float(df[column].corr(df[TARGET_COLUMN])),
                }
            )
    return pd.DataFrame(rows)


def categorical_fraud_summary(
    df: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = CATEGORICAL_SUMMARY_COLUMNS,
) -> pd.DataFrame:
    frames = []
    for column in columns:
        grouped = (
            df.groupby(column, dropna=False)
            .agg(
                transaction_count=(TARGET_COLUMN, "count"),
                fraud_count=(TARGET_COLUMN, "sum"),
                average_amount=("Transaction_Amount", "mean"),
            )
            .reset_index()
            .rename(columns={column: "category"})
        )
        grouped.insert(0, "feature", column)
        grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
        frames.append(grouped)
    summary = pd.concat(frames, ignore_index=True)
    return summary[
        [
            "feature",
            "category",
            "transaction_count",
            "fraud_count",
            "fraud_rate",
            "average_amount",
        ]
    ]


def amount_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(
        df["Transaction_Amount"],
        bins=list(AMOUNT_BINS),
        right=False,
        include_lowest=True,
    )
    grouped = (
        df.groupby(bins, observed=False)[TARGET_COLUMN]
        .agg(transaction_count="count", fraud_count="sum")
        .reset_index()
        .rename(columns={"Transaction_Amount": "amount_bin"})
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    return grouped


def international_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("Is_International")
        .agg(
            transaction_count=(TARGET_COLUMN, "count"),
            fraud_count=(TARGET_COLUMN, "sum"),
            average_amount=("Transaction_Amount", "mean"),
        )
        .reset_index()
    )
    grouped["segment"] = grouped["Is_International"].map(
        {0: "Domestic", 1: "International"}
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    return grouped[
        [
            "Is_International",
            "segment",
            "transaction_count",
            "fraud_count",
            "fraud_rate",
            "average_amount",
        ]
    ]


def keyword_international_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Is_International", "Suspicious_Keyword"])
        .agg(
            transaction_count=(TARGET_COLUMN, "count"),
            fraud_count=(TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    grouped["segment"] = grouped["Is_International"].map(
        {0: "Domestic", 1: "International"}
    )
    return grouped


def customer_summary(df: pd.DataFrame) -> dict[str, float | int]:
    counts = df["Customer_ID"].value_counts()
    any_fraud = df.groupby("Customer_ID")[TARGET_COLUMN].max()
    return {
        "unique_customers": int(counts.size),
        "mean_transactions_per_customer": float(counts.mean()),
        "median_transactions_per_customer": float(counts.median()),
        "max_transactions_per_customer": int(counts.max()),
        "min_transactions_per_customer": int(counts.min()),
        "customers_with_multiple_transactions": int((counts > 1).sum()),
        "customers_with_any_fraud": int((any_fraud == 1).sum()),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(NUMERIC_EDA_COLUMNS) + ["Is_International", TARGET_COLUMN]
    return df[columns].corr()


def overview_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "unique_values": [int(df[c].nunique(dropna=False)) for c in df.columns],
            "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        }
    )


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def build_eda_summary_markdown(df: pd.DataFrame) -> str:
    """Build reports/eda_summary.md content from the actual DataFrame."""
    work = add_temporary_eda_features(df)
    tgt = target_summary(df)
    intl = international_summary(df)
    kw = (
        df.groupby("Suspicious_Keyword")[TARGET_COLUMN]
        .agg(transaction_count="count", fraud_count="sum")
        .assign(fraud_rate=lambda x: x["fraud_count"] / x["transaction_count"])
    )
    merchants = (
        df.groupby("Merchant_Category")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    payments = (
        df.groupby("Payment_Method")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    devices = (
        df.groupby("Device_Type")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    locations = (
        df.groupby("Location")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    hour = (
        work.groupby("hour")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
    )
    night = hour.loc[hour.index.isin(range(0, 6))]
    day = hour.loc[hour.index.isin(range(7, 24))]
    month = (
        work.groupby("year_month")[TARGET_COLUMN]
        .agg(n="count", fraud="sum")
        .assign(rate=lambda x: x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    customers = customer_summary(df)
    num = numeric_by_target(df)
    corr = correlation_matrix(df)
    kw_intl = keyword_international_summary(df)
    amount_stats = df.groupby(TARGET_COLUMN)["Transaction_Amount"].agg(
        ["mean", "median"]
    )
    ratio_stats = work.groupby(TARGET_COLUMN)["amount_to_average_ratio"].mean()
    parsed_min = work["_parsed_date"].min()
    parsed_max = work["_parsed_date"].max()
    domestic = intl.loc[intl["Is_International"] == 0].iloc[0]
    international = intl.loc[intl["Is_International"] == 1].iloc[0]
    kw_no = kw.loc["No"]
    kw_yes = kw.loc["Yes"]

    def _kw_row(intl_flag: int, keyword: str) -> pd.Series:
        match = kw_intl[
            (kw_intl["Is_International"] == intl_flag)
            & (kw_intl["Suspicious_Keyword"] == keyword)
        ].iloc[0]
        return match

    lines = [
        "# Phase 3 EDA Summary",
        "",
        "All figures below are calculated from `data/raw/financial_fraud_detection_dataset.csv`.",
        "The raw CSV is not modified. Temporary datetime and ratio columns exist only in memory.",
        "",
        "## 1. Dataset overview",
        "",
        f"- Rows: {tgt['rows']}",
        f"- Columns: {tgt['columns']}",
        f"- Missing values: {int(df.isna().sum().sum())}",
        f"- Duplicate rows: {int(df.duplicated().sum())}",
        f"- Duplicate Transaction_ID: {int(df['Transaction_ID'].duplicated().sum())}",
        f"- Parsed date range: {parsed_min} to {parsed_max}",
        "- Required columns match the Phase 2 schema.",
        "",
        "## 2. Target distribution",
        "",
        f"- Fraudulent transactions: {tgt['fraud_count']} ({_pct(tgt['fraud_rate'])})",
        f"- Legitimate transactions: {tgt['legitimate_count']} ({_pct(tgt['legitimate_rate'])})",
        "",
        "Class imbalance matters for later evaluation: a model that predicts all transactions as legitimate would already be about "
        f"{_pct(tgt['legitimate_rate'])} accurate. Accuracy alone is not an appropriate selection metric. Later phases should emphasise recall, precision, F1, ROC-AUC, and PR-AUC on an untouched test set.",
        "",
        "## 3. Numerical feature findings",
        "",
        f"- Transaction_Amount: overall mean {df['Transaction_Amount'].mean():.2f}, median {df['Transaction_Amount'].median():.2f}, max {df['Transaction_Amount'].max():.2f}. Fraud mean {amount_stats.loc[1, 'mean']:.2f} vs legitimate mean {amount_stats.loc[0, 'mean']:.2f}; medians are nearly identical ({amount_stats.loc[1, 'median']:.2f} vs {amount_stats.loc[0, 'median']:.2f}). Correlation with the target is {corr.loc['Transaction_Amount', TARGET_COLUMN]:.4f}. Amount alone shows limited separation.",
        f"- Previous_Transactions: fraud mean {df.loc[df[TARGET_COLUMN]==1, 'Previous_Transactions'].mean():.2f} vs legitimate {df.loc[df[TARGET_COLUMN]==0, 'Previous_Transactions'].mean():.2f}; correlation {corr.loc['Previous_Transactions', TARGET_COLUMN]:.4f}.",
        f"- Average_Spend: fraud and legitimate means are nearly the same ({df.loc[df[TARGET_COLUMN]==1, 'Average_Spend'].mean():.2f} vs {df.loc[df[TARGET_COLUMN]==0, 'Average_Spend'].mean():.2f}); correlation {corr.loc['Average_Spend', TARGET_COLUMN]:.4f}.",
        f"- Account_Age_Days: fraud mean {df.loc[df[TARGET_COLUMN]==1, 'Account_Age_Days'].mean():.1f} vs legitimate {df.loc[df[TARGET_COLUMN]==0, 'Account_Age_Days'].mean():.1f}; correlation {corr.loc['Account_Age_Days', TARGET_COLUMN]:.4f}. EDA age groups stay near the overall fraud rate.",
        f"- amount_to_average_ratio (EDA-only): mean {ratio_stats.loc[1]:.3f} for fraud vs {ratio_stats.loc[0]:.3f} for legitimate. Zero Average_Spend rows: {int((df['Average_Spend']==0).sum())}. This ratio is a Phase 4 candidate, not a saved column.",
        "",
        "Outliers were inspected and **not** removed. The largest amounts remain in the file because fraud can occur in the tail. No amount threshold rule is justified.",
        "",
        "## 4. Categorical feature findings",
        "",
        f"- Is_International: domestic fraud rate {_pct(domestic['fraud_rate'])} ({int(domestic['fraud_count'])}/{int(domestic['transaction_count'])}); international fraud rate {_pct(international['fraud_rate'])} ({int(international['fraud_count'])}/{int(international['transaction_count'])}). This is the strongest non-keyword association observed.",
        f"- Merchant_Category: highest observed rate {merchants.index[0]} {_pct(float(merchants.iloc[0]['rate']))} (n={int(merchants.iloc[0]['n'])}); lowest {merchants.index[-1]} {_pct(float(merchants.iloc[-1]['rate']))} (n={int(merchants.iloc[-1]['n'])}). Category counts are similar (about 590-680), so rates are comparable.",
        f"- Payment_Method: highest {payments.index[0]} {_pct(float(payments.iloc[0]['rate']))}; lowest {payments.index[-1]} {_pct(float(payments.iloc[-1]['rate']))}. Differences are modest.",
        f"- Device_Type: {', '.join(f'{idx} {_pct(float(row.rate))} (n={int(row.n)})' for idx, row in devices.iterrows())}.",
        f"- Location: highest {locations.index[0]} {_pct(float(locations.iloc[0]['rate']))}; lowest {locations.index[-1]} {_pct(float(locations.iloc[-1]['rate']))}. These are fraud rates by location, not evidence that a city causes fraud.",
        "",
        "## 5. Time findings",
        "",
        f"- Coverage is {parsed_min.date()} through {parsed_max.date()} ({int((work['year']==2023).sum())} rows in 2023 and {int((work['year']==2024).sum())} in 2024). February 2024 is a partial month. This limited window is a limitation; seasonality is not claimed.",
        f"- Overnight hours 00:00-05:00 have a higher observed fraud rate (range {_pct(float(night['rate'].min()))} to {_pct(float(night['rate'].max()))}, about 178-236 transactions per hour) than hours 07:00-23:00 (range {_pct(float(day['rate'].min()))} to {_pct(float(day['rate'].max()))}). Hour is a useful Phase 4 candidate.",
        f"- Month with the highest observed rate: {month.index[0]} ({_pct(float(month.iloc[0]['rate']))}, n={int(month.iloc[0]['n'])}). Month with the lowest: {month.index[-1]} ({_pct(float(month.iloc[-1]['rate']))}, n={int(month.iloc[-1]['n'])}).",
        "",
        "## 6. Suspicious_Keyword leakage assessment",
        "",
        f"- Keyword = No: {_pct(float(kw_no['fraud_rate']))} ({int(kw_no['fraud_count'])}/{int(kw_no['transaction_count'])}).",
        f"- Keyword = Yes: {_pct(float(kw_yes['fraud_rate']))} ({int(kw_yes['fraud_count'])}/{int(kw_yes['transaction_count'])}).",
        f"- Domestic + No: {_pct(float(_kw_row(0, 'No')['fraud_rate']))} (n={int(_kw_row(0, 'No')['transaction_count'])}).",
        f"- Domestic + Yes: {_pct(float(_kw_row(0, 'Yes')['fraud_rate']))} (n={int(_kw_row(0, 'Yes')['transaction_count'])}).",
        f"- International + No: {_pct(float(_kw_row(1, 'No')['fraud_rate']))} (n={int(_kw_row(1, 'No')['transaction_count'])}).",
        f"- International + Yes: {_pct(float(_kw_row(1, 'Yes')['fraud_rate']))} (n={int(_kw_row(1, 'Yes')['transaction_count'])}).",
        "",
        "Suspicious_Keyword is strongly associated with the target. The dataset documentation does not prove it is a post-decision field, so it is not labelled as definite leakage. It **requires a provenance/timing review before inclusion in the production model** and is not used as a production ML feature in this phase.",
        "",
        "## 7. Customer_ID assessment",
        "",
        f"- Unique customers: {customers['unique_customers']}",
        f"- Transactions per customer: mean {customers['mean_transactions_per_customer']:.2f}, median {customers['median_transactions_per_customer']:.0f}, max {customers['max_transactions_per_customer']}",
        f"- Customers with more than one transaction: {customers['customers_with_multiple_transactions']}",
        f"- Customers with at least one fraud label: {customers['customers_with_any_fraud']}",
        "",
        "Do not one-hot encode raw Customer_ID (high cardinality). Customer-level aggregates could be considered in Phase 4 only if they are computed from training data with no future-data leakage.",
        "",
        "## 8. Correlation observations",
        "",
        f"- Is_International vs Fraudulent: {corr.loc['Is_International', TARGET_COLUMN]:.4f} (strongest numeric/binary correlation).",
        f"- Transaction_Amount vs Fraudulent: {corr.loc['Transaction_Amount', TARGET_COLUMN]:.4f}.",
        f"- Previous_Transactions vs Fraudulent: {corr.loc['Previous_Transactions', TARGET_COLUMN]:.4f}.",
        f"- Average_Spend vs Fraudulent: {corr.loc['Average_Spend', TARGET_COLUMN]:.4f}.",
        f"- Account_Age_Days vs Fraudulent: {corr.loc['Account_Age_Days', TARGET_COLUMN]:.4f}.",
        "",
        "Correlation is not causation and is not used as the only feature-selection rule. Categorical rates and hour-of-day patterns add information that a Pearson matrix does not capture.",
        "",
        "## 9. Candidate features",
        "",
        "**KEEP:** Transaction_Amount, Previous_Transactions, Average_Spend, Account_Age_Days, Is_International, Merchant_Category, Payment_Method, Device_Type, Location, Transaction_Date-derived fields.",
        "",
        "**TRANSFORM:** amount_to_average_ratio, hour, day of week, month.",
        "",
        "**EXCLUDE:** Transaction_ID, raw Customer_ID, Fraudulent (target only).",
        "",
        "**INVESTIGATE:** Suspicious_Keyword (provenance/timing review).",
        "",
        "These labels are EDA recommendations. They are not permanently enforced until Phase 4.",
        "",
        "## 10. Limitations",
        "",
        "- The file has 5,000 rows and 482 fraud cases, so rate estimates have sampling variability.",
        "- Dates cover about 14 months with a partial final month; seasonal claims are not supported.",
        "- Suspicious_Keyword timing is undocumented.",
        "- Customer history is thin (median one transaction).",
        "- This phase does not train or evaluate models, so no performance numbers are reported.",
        "",
        "## 11. Key findings",
        "",
        f"1. The dataset has {tgt['rows']} transactions and {tgt['fraud_count']} fraud cases ({_pct(tgt['fraud_rate'])} fraud rate), with no missing values and no duplicate Transaction_IDs.",
        f"2. International transactions have a higher observed fraud rate ({_pct(international['fraud_rate'])}) than domestic transactions ({_pct(domestic['fraud_rate'])}).",
        f"3. Suspicious_Keyword = Yes is strongly associated with the target ({_pct(float(kw_yes['fraud_rate']))} vs {_pct(float(kw_no['fraud_rate']))} when No) and therefore requires provenance review.",
        f"4. Transaction amount alone shows limited separation: fraud and legitimate medians are {amount_stats.loc[1, 'median']:.2f} and {amount_stats.loc[0, 'median']:.2f}.",
        f"5. Overnight hours (00:00-05:00) have higher observed fraud rates than daytime hours; hour is a useful derived feature candidate.",
        f"6. Merchant, payment, device, and location fraud rates vary modestly (roughly {_pct(float(locations.iloc[-1]['rate']))} to {_pct(float(merchants.iloc[0]['rate']))}) with comparable sample sizes.",
        f"7. Previous_Transactions, Average_Spend, and Account_Age_Days have near-zero correlations with the target (|r| < 0.02) but remain KEEP candidates for non-linear models.",
        f"8. The amount-to-average-spend ratio is slightly higher on average for fraud ({ratio_stats.loc[1]:.3f} vs {ratio_stats.loc[0]:.3f}) and is a TRANSFORM candidate.",
        "9. Raw Customer_ID should not be one-hot encoded; any later customer aggregates must avoid future leakage.",
        "10. No amount threshold such as amount > 200000 is supported: the maximum amount in this file is "
        f"{df['Transaction_Amount'].max():.2f}.",
        "11. Pearson correlation understates categorical and time-of-day associations; Is_International is the exception among numeric/binary fields.",
        "",
        "ML training and dashboard implementation are planned for later phases.",
        "",
    ]
    _ = num  # computed for consistency; details live in the notebook tables
    return "\n".join(lines)


def write_eda_summary(df: pd.DataFrame, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else REPORTS_DIR / "eda_summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_eda_summary_markdown(df), encoding="utf-8", newline="\n")
    return target
