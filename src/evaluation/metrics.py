"""Compute leakage-safe test metrics from saved pipelines.

Metrics are calculated from actual predictions. Nothing is hard-coded.
The default classification threshold is used. Threshold tuning is Phase 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Return scalar metrics and confusion counts for a binary fraud label."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype="float64")
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    metrics = {
        "Accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "Precision": float(precision_score(y_true_arr, y_pred_arr, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred_arr, average="binary", zero_division=0)),
        "F1": float(f1_score(y_true_arr, y_pred_arr, average="binary", zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "support": int(len(y_true_arr)),
    }
    if y_proba_arr.ndim != 1:
        raise ValueError("y_proba must be the positive-class probability vector.")
    if np.unique(y_true_arr).size < 2:
        metrics["ROC_AUC"] = float("nan")
    else:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true_arr, y_proba_arr))
    metrics["PR_AUC"] = float(average_precision_score(y_true_arr, y_proba_arr))
    return metrics


def dummy_always_legitimate(y_true: pd.Series | np.ndarray) -> dict[str, Any]:
    """Constant 'all legitimate' baseline. Not a trained model."""
    n = len(y_true)
    y_pred = np.zeros(n, dtype=int)
    y_proba = np.zeros(n, dtype="float64")
    return compute_binary_metrics(y_true, y_pred, y_proba)


def roc_points(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_proba, dtype="float64"))
    return fpr, tpr, thresholds


def pr_points(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true).astype(int),
        np.asarray(y_proba, dtype="float64"),
    )
    return precision, recall, thresholds
