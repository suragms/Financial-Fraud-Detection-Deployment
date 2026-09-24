"""Read-only data helpers for the Streamlit dashboard.

These functions never write to the raw CSV, never retrain, and never fit
preprocessing, calibration, or a new threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.eda import raw_dataset_md5
from src.data.loader import DATE_FORMAT, TARGET_COLUMN, load_fraud_dataset
from src.features.feature_engineering import FEATURE_INPUT_COLUMNS
from src.models.predict import (
    PRODUCTION_METADATA_PATH,
    PRODUCTION_PIPELINE_PATH,
    load_production_pipeline,
    predict_records,
)
from src.utils.paths import RAW_DATASET_PATH, REPORTS_DIR

EXPECTED_RAW_DATASET_MD5 = "9a4a90ce2e07a717b4289dc95a71663c"
EXPECTED_RAW_DATASET_MD5_LF = "9252ebcb3c7684dee5d1a8b30de1f974"
VALID_RAW_DATASET_MD5S = (EXPECTED_RAW_DATASET_MD5, EXPECTED_RAW_DATASET_MD5_LF)
EXPECTED_DATASET_ROWS = 5000

DISPLAY_COLUMNS = [
    "Transaction_ID",
    "Transaction_Date",
    "Transaction_Amount",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "predicted_probability",
    "risk_score",
    "risk_band",
    "predicted_label",
    "prediction_text",
    "Historical Ground Truth",
]

# Phase 6 test-set dummy, documented in reports/model_comparison.md.
PHASE6_DUMMY_ACCURACY = 0.9040
PHASE6_DUMMY_RECALL = 0.0

FORBIDDEN_MODEL_INPUTS = (
    "Fraudulent",
    "Customer_ID",
    "Transaction_ID",
    "Suspicious_Keyword",
)


def load_raw_dataset() -> pd.DataFrame:
    """Load the primary CSV without modifying it."""
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            "The dataset was not found at data/raw/financial_fraud_detection_dataset.csv."
        )
    frame = load_fraud_dataset()
    if len(frame) != EXPECTED_DATASET_ROWS:
        raise ValueError(f"Expected {EXPECTED_DATASET_ROWS} rows, found {len(frame)}.")
    return frame


def load_metadata() -> dict[str, Any]:
    if not PRODUCTION_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Production metadata was not found. Run `python run_phase7.py` first."
        )
    payload = json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))
    threshold = float(payload.get("production_threshold", -1))
    if not 0.0 < threshold < 1.0:
        raise ValueError("Production threshold in metadata is not between 0 and 1.")
    return payload


def load_pipeline():
    if not PRODUCTION_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Production model was not found at models/production/final_fraud_pipeline.joblib."
        )
    return load_production_pipeline()


def load_final_test_metrics() -> dict[str, Any]:
    """Frozen Phase 7 one-shot test metrics from disk. Not recomputed."""
    path = REPORTS_DIR / "final_model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError("reports/final_model_metrics.csv is missing.")
    row = pd.read_csv(path).iloc[0].to_dict()
    return row


def category_options(frame: pd.DataFrame, column: str) -> list[str]:
    values = frame[column].dropna().astype(str).sort_values().unique().tolist()
    return values


def with_parsed_hour(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = pd.to_datetime(out["Transaction_Date"], format=DATE_FORMAT, errors="coerce")
    out["transaction_hour"] = parsed.dt.hour
    out["origin"] = out["Is_International"].map({0: "Domestic", 1: "International"})
    out["historical_label"] = out[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraud"})
    return out


def score_dataset(raw: pd.DataFrame | None = None, pipeline=None) -> pd.DataFrame:
    """Score the historical dataset with the frozen production pipeline.

    Historical ``Fraudulent`` is copied only as ground truth for analytics.
    It is not passed into the model.
    """
    frame = raw if raw is not None else load_raw_dataset()
    model = pipeline if pipeline is not None else load_pipeline()
    model_inputs = frame.loc[:, list(FEATURE_INPUT_COLUMNS)].copy()
    leaked = [column for column in FORBIDDEN_MODEL_INPUTS if column in model_inputs.columns]
    if leaked:
        raise ValueError("Forbidden columns reached model inputs: " + ", ".join(leaked))
    scored = predict_records(model_inputs, pipeline=model)
    display = frame.loc[
        :,
        [
            "Transaction_ID",
            "Transaction_Date",
            "Transaction_Amount",
            "Merchant_Category",
            "Payment_Method",
            "Device_Type",
            "Location",
            "Is_International",
            TARGET_COLUMN,
        ],
    ].reset_index(drop=True)
    display = display.rename(columns={TARGET_COLUMN: "Historical Ground Truth"})
    combined = pd.concat([display, scored], axis=1)
    combined["prediction_text"] = combined["predicted_label"].map(
        {1: "Flag for review", 0: "Legitimate"}
    )
    combined["origin"] = combined["Is_International"].map({0: "Domestic", 1: "International"})
    return combined


def fraud_rate_table(frame: pd.DataFrame, column: str, min_n: int = 1) -> pd.DataFrame:
    if "Fraudulent" not in frame.columns:
        raise ValueError("fraud_rate_table expects the historical Fraudulent column.")
    grouped = (
        frame.groupby(column, dropna=False)["Fraudulent"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"size": "n", "sum": "frauds", "mean": "fraud_rate"})
    )
    grouped = grouped[grouped["n"] >= min_n].sort_values("fraud_rate", ascending=False)
    grouped["label"] = grouped[column].astype(str) + " (n=" + grouped["n"].astype(int).astype(str) + ")"
    return grouped


def dataset_hash() -> str:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            "The historical dataset was not found at data/raw/financial_fraud_detection_dataset.csv."
        )
    digest = raw_dataset_md5()
    if digest == EXPECTED_RAW_DATASET_MD5_LF:
        return EXPECTED_RAW_DATASET_MD5
    return digest


def export_scored_csv(scored: pd.DataFrame, path: Path | str) -> Path:
    """Write scored rows to a new file. Never writes the raw dataset."""
    target = Path(path)
    if target.resolve() == RAW_DATASET_PATH.resolve():
        raise ValueError("Refusing to overwrite the raw dataset.")
    columns = [column for column in DISPLAY_COLUMNS if column in scored.columns]
    target.parent.mkdir(parents=True, exist_ok=True)
    scored.loc[:, columns].to_csv(target, index=False)
    return target
