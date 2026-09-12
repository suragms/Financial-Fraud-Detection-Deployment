"""Leakage-safe inference for the Phase 7 production pipeline.

Accepts a raw transaction record or an already-engineered feature frame.
Does not retrain. Does not use Customer_ID, Suspicious_Keyword, or the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.loader import ValidationError
from src.features.feature_engineering import (
    EXCLUDED_FEATURES,
    FEATURE_INPUT_COLUMNS,
    MODEL_FEATURES,
    build_features,
)
from src.models.scoring import FraudRiskPipeline, VALID_RISK_BAND_NAMES
from src.utils.paths import PRODUCTION_MODELS_DIR

PRODUCTION_PIPELINE_PATH = PRODUCTION_MODELS_DIR / "final_fraud_pipeline.joblib"
PRODUCTION_METADATA_PATH = PRODUCTION_MODELS_DIR / "model_metadata.json"

_FORBIDDEN_MODEL_COLUMNS = set(EXCLUDED_FEATURES)

_RAW_NUMERIC_COLUMNS = (
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
)


def _validate_raw_inference_values(frame: pd.DataFrame) -> None:
    """Reject invalid raw values with user-facing errors before scoring."""
    for column in _RAW_NUMERIC_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy(dtype="float64")
        if numeric.isna().any() or not np.isfinite(values).all():
            raise ValidationError(
                f"{column} must be a finite number. Missing, text, NaN, or infinite values are not allowed."
            )
        if (values < 0).any():
            raise ValidationError(f"{column} cannot be negative.")
    international = pd.to_numeric(frame["Is_International"], errors="coerce")
    if international.isna().any() or not set(international.dropna().unique()).issubset({0, 1}):
        raise ValidationError("Is_International must be 0 (domestic) or 1 (international).")


def load_production_pipeline(path: Path | None = None) -> FraudRiskPipeline:
    """Load the saved production wrapper. Does not fit or calibrate."""
    target = path or PRODUCTION_PIPELINE_PATH
    if not target.exists():
        raise FileNotFoundError(
            "Production pipeline was not found at "
            f"{target.as_posix()}. Run `python run_phase7.py` first."
        )
    pipeline = joblib.load(target)
    if not isinstance(pipeline, FraudRiskPipeline):
        raise TypeError(
            f"Expected FraudRiskPipeline, got {type(pipeline).__name__}."
        )
    return pipeline


def _as_frame(record: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
    if isinstance(record, pd.DataFrame):
        return record.copy()
    if isinstance(record, pd.Series):
        return record.to_frame().T
    if isinstance(record, dict):
        return pd.DataFrame([record])
    raise TypeError(
        "Inference input must be a dict, pandas Series, or pandas DataFrame. "
        f"Got {type(record).__name__}."
    )


def prepare_features(record: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Return a MODEL_FEATURES frame. Raw dates are engineered; they are not model inputs."""
    frame = _as_frame(record)
    if not len(frame):
        raise ValidationError("Inference input is empty.")

    leaked = [column for column in frame.columns if column in _FORBIDDEN_MODEL_COLUMNS]
    engineered_ready = all(column in frame.columns for column in MODEL_FEATURES)
    raw_ready = all(column in frame.columns for column in FEATURE_INPUT_COLUMNS)

    if engineered_ready:
        features = frame.loc[:, list(MODEL_FEATURES)].copy()
        still_forbidden = [column for column in features.columns if column in _FORBIDDEN_MODEL_COLUMNS]
        if still_forbidden:
            raise ValidationError(
                "Forbidden columns reached the model matrix: " + ", ".join(still_forbidden)
            )
        return features

    if raw_ready:
        _validate_raw_inference_values(frame)
        return build_features(frame.loc[:, list(FEATURE_INPUT_COLUMNS)])

    missing_raw = [column for column in FEATURE_INPUT_COLUMNS if column not in frame.columns]
    missing_model = [column for column in MODEL_FEATURES if column not in frame.columns]
    raise ValidationError(
        "Inference is missing required fields. Provide either the raw transaction "
        f"columns {list(FEATURE_INPUT_COLUMNS)} (missing: {missing_raw}) or the "
        f"engineered MODEL_FEATURES (missing: {missing_model}). "
        f"Non-feature columns present but ignored for modeling: {leaked}."
    )


def predict_records(
    record: pd.DataFrame | pd.Series | dict[str, Any],
    *,
    pipeline: FraudRiskPipeline | None = None,
) -> pd.DataFrame:
    """Score one or more transactions with the saved production pipeline."""
    model = pipeline or load_production_pipeline()
    features = prepare_features(record)
    scored = model.score(features)
    if not np.isfinite(scored["predicted_probability"].to_numpy()).all():
        raise ValueError("Production pipeline produced a non-finite probability.")
    if not scored["risk_band"].isin(VALID_RISK_BAND_NAMES).all():
        raise ValueError("Production pipeline produced an unknown risk band.")
    return scored.reset_index(drop=True)


def predict_transaction(
    record: pd.DataFrame | pd.Series | dict[str, Any],
    *,
    pipeline: FraudRiskPipeline | None = None,
) -> dict[str, Any]:
    """Score a single transaction and return probability, risk score, band, and label."""
    frame = _as_frame(record)
    if len(frame) != 1:
        raise ValidationError(
            f"predict_transaction expects exactly one row. Received {len(frame)}."
        )
    row = predict_records(frame, pipeline=pipeline).iloc[0]
    return {
        "predicted_probability": float(row["predicted_probability"]),
        "risk_score": float(row["risk_score"]),
        "risk_band": str(row["risk_band"]),
        "predicted_label": int(row["predicted_label"]),
    }
