"""Production scoring helpers: threshold metrics, risk bands, inference wrapper.

The wrapper stores a fitted estimator plus a frozen operating threshold.
It does not retrain. Risk scores are model indicators, not proof of fraud.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


THRESHOLD_GRID = tuple(round(float(value), 2) for value in np.arange(0.10, 0.91, 0.05))

# Inclusive lower bound, exclusive upper bound except the last band includes 100.
RISK_BANDS: tuple[dict[str, Any], ...] = (
    {"name": "Low Risk", "min_inclusive": 0.0, "max_exclusive": 30.0, "range": "0-29"},
    {"name": "Medium Risk", "min_inclusive": 30.0, "max_exclusive": 60.0, "range": "30-59"},
    {"name": "High Risk", "min_inclusive": 60.0, "max_exclusive": 80.0, "range": "60-79"},
    {"name": "Critical Risk", "min_inclusive": 80.0, "max_exclusive": 100.01, "range": "80-100"},
)

VALID_RISK_BAND_NAMES = tuple(band["name"] for band in RISK_BANDS)

MIN_RECALL_PRIMARY = 0.70
MAX_FPR_PRIMARY = 0.35
MIN_RECALL_FALLBACK = 0.65
MAX_FPR_FALLBACK = 0.40


def assign_risk_band(
    risk_score: float | np.ndarray | pd.Series,
    bands: tuple[dict[str, Any], ...] = RISK_BANDS,
) -> np.ndarray | str:
    """Map a 0-100 risk score onto the provided bands."""
    values = np.asarray(risk_score, dtype="float64")
    scalar = values.ndim == 0
    flat = np.atleast_1d(values)
    labels = np.empty(flat.shape, dtype=object)
    labels[:] = bands[0]["name"]
    for band in bands:
        mask = (flat >= float(band["min_inclusive"])) & (flat < float(band["max_exclusive"]))
        labels[mask] = band["name"]
    labels[flat >= 100.0] = bands[-1]["name"]
    if scalar:
        return str(labels[0])
    return labels


def probability_to_risk_score(probability: np.ndarray | pd.Series) -> np.ndarray:
    score = np.asarray(probability, dtype="float64") * 100.0
    return np.clip(score, 0.0, 100.0)


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype="float64")
    y_pred = (y_proba_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    n_neg = int(tn + fp)
    n_pos = int(tp + fn)
    flagged = int(tp + fp)
    n = int(len(y_true_arr))
    return {
        "threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "Precision": float(precision_score(y_true_arr, y_pred, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred, average="binary", zero_division=0)),
        "F1": float(f1_score(y_true_arr, y_pred, average="binary", zero_division=0)),
        "False_Positive_Rate": float(fp / n_neg) if n_neg else 0.0,
        "False_Negative_Rate": float(fn / n_pos) if n_pos else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "flagged": flagged,
        "flagged_rate": float(flagged / n) if n else 0.0,
        "legitimate_predicted": int(tn + fn),
        "legitimate_predicted_rate": float((tn + fn) / n) if n else 0.0,
        "support": n,
    }


def threshold_table(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    rows = [metrics_at_threshold(y_true, y_proba, threshold) for threshold in THRESHOLD_GRID]
    return pd.DataFrame(rows)


def select_operating_threshold(table: pd.DataFrame) -> pd.Series:
    """Pick a fraud-review threshold from a training-only metric table.

    Primary screen: recall at least 0.70 and FPR at most 0.35.
    Fallback: recall at least 0.65 and FPR at most 0.40.
    Among eligible rows maximize F1, then recall, then precision, then lower FPR.
    Accuracy is not the objective.
    """
    if table.empty:
        raise ValueError("Threshold table is empty.")
    eligible = table[
        (table["Recall"] >= MIN_RECALL_PRIMARY) & (table["False_Positive_Rate"] <= MAX_FPR_PRIMARY)
    ]
    rule = (
        "recall at least 0.70 and FPR at most 0.35, "
        "then maximize F1 / recall / precision and minimize FPR"
    )
    if eligible.empty:
        eligible = table[
            (table["Recall"] >= MIN_RECALL_FALLBACK)
            & (table["False_Positive_Rate"] <= MAX_FPR_FALLBACK)
        ]
        rule = (
            "fallback: recall at least 0.65 and FPR at most 0.40, "
            "then maximize F1 / recall / precision and minimize FPR"
        )
    if eligible.empty:
        eligible = table.copy()
        rule = "no row met recall/FPR screens; maximize F1 among all training thresholds"
    ranked = eligible.sort_values(
        by=["F1", "Recall", "Precision", "False_Positive_Rate", "threshold"],
        ascending=[False, False, False, True, True],
    )
    chosen = ranked.iloc[0].copy()
    chosen["selection_rule"] = rule
    return chosen


class FraudRiskPipeline:
    """Fitted estimator + frozen threshold + risk-score mapping for inference."""

    def __init__(
        self,
        estimator: Any,
        *,
        threshold: float,
        model_name: str,
        strategy: str,
        calibration_method: str,
        risk_bands: tuple[dict[str, Any], ...] = RISK_BANDS,
    ) -> None:
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"Production threshold must be in (0, 1). Got {threshold}.")
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("Estimator must implement predict_proba.")
        self.estimator = estimator
        self.threshold = float(threshold)
        self.model_name = model_name
        self.strategy = strategy
        self.calibration_method = calibration_method
        self.risk_bands = tuple(risk_bands)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X), dtype="float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)[:, 1]
        risk = probability_to_risk_score(proba)
        bands = assign_risk_band(risk, self.risk_bands)
        labels = (proba >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "predicted_probability": proba,
                "risk_score": risk,
                "risk_band": bands,
                "predicted_label": labels,
            },
            index=getattr(X, "index", None),
        )
