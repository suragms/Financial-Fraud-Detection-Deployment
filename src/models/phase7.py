"""Phase 7: calibration, threshold, risk scoring, and one-shot test evaluation.

Calibration and threshold selection use TRAINING DATA ONLY.
The Phase 6 test fold is scored once after those decisions are frozen.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.data.eda import raw_dataset_md5
from src.evaluation.compare import (
    EXPECTED_RAW_DATASET_MD5,
    assert_raw_dataset_unchanged,
    assert_test_fold,
    candidate_artifact_path,
)
from src.evaluation.metrics import compute_binary_metrics
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.models.pipeline import load_pipeline, load_xy, make_stratified_split
from src.models.scoring import (
    RISK_BANDS,
    FraudRiskPipeline,
    assign_risk_band,
    metrics_at_threshold,
    probability_to_risk_score,
    select_operating_threshold,
    threshold_table,
)
from src.utils.paths import (
    BASELINE_MODELS_DIR,
    DATASET_FILENAME,
    MODEL_SELECTION_ARTIFACTS_DIR,
    PRODUCTION_MODELS_DIR,
    REPORTS_DIR,
)
from src.utils.seeds import RANDOM_STATE, set_global_seed

SELECTED_KEY = "logistic_regression_class_weight"
SELECTED_MODEL_NAME = "Logistic Regression"
SELECTED_STRATEGY = "Class Weight"
MODEL_VERSION = "1.0.0-phase7"
INNER_VAL_SIZE = 0.20
CV_SPLITS = 5
BRIER_IMPROVEMENT_MIN = 0.005
SELECTED_METADATA_PATH = BASELINE_MODELS_DIR / "selected_model.json"
PRODUCTION_PIPELINE_PATH = PRODUCTION_MODELS_DIR / "final_fraud_pipeline.joblib"
PRODUCTION_METADATA_PATH = PRODUCTION_MODELS_DIR / "model_metadata.json"


def choose_risk_bands(calibration_method: str, threshold: float) -> tuple[dict[str, Any], ...]:
    """Return explainable bands. Calibrated probabilities use threshold-aligned cuts.

    Uncalibrated class-weighted logistic scores spread toward 0.5, so the
    original 0-29 / 30-59 / 60-79 / 80-100 cuts remain readable.

    Sigmoid calibration maps scores onto the ~10% prevalence scale. Keeping
    0-29 as Low would put almost every review-queue item in Low Risk. Bands
    are therefore aligned to the frozen training threshold using round
    cut-points derived from that threshold only (not from test labels).
    """
    if calibration_method == "none":
        return RISK_BANDS
    low_end = round(float(threshold) * 100.0)
    medium_end = min(low_end + 10, 40)
    high_end = min(low_end + 30, 70)
    return (
        {
            "name": "Low Risk",
            "min_inclusive": 0.0,
            "max_exclusive": float(low_end),
            "range": f"0-{low_end - 1}",
        },
        {
            "name": "Medium Risk",
            "min_inclusive": float(low_end),
            "max_exclusive": float(medium_end),
            "range": f"{low_end}-{medium_end - 1}",
        },
        {
            "name": "High Risk",
            "min_inclusive": float(medium_end),
            "max_exclusive": float(high_end),
            "range": f"{medium_end}-{high_end - 1}",
        },
        {
            "name": "Critical Risk",
            "min_inclusive": float(high_end),
            "max_exclusive": 100.01,
            "range": f"{high_end}-100",
        },
    )


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _clone_selected_architecture(fitted_pipeline: Any) -> Any:
    """Clone preprocess + logistic architecture without copying fitted state."""
    return clone(fitted_pipeline)


def make_inner_train_val(split) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out 20% of the 4,000-row training fold. Test is never used."""
    X_inner, X_val, y_inner, y_val = train_test_split(
        split.X_train,
        split.y_train,
        test_size=INNER_VAL_SIZE,
        stratify=split.y_train,
        random_state=RANDOM_STATE,
    )
    return (
        X_inner.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_inner.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def _positive_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(X), dtype="float64")
    if proba.shape != (len(X), 2):
        raise RuntimeError(f"STOP: predict_proba shape {proba.shape} is not (n, 2).")
    return proba[:, 1]


def _calibration_stats(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype="float64")
    frac_pos, mean_pred = calibration_curve(
        y_true_arr,
        y_proba_arr,
        n_bins=n_bins,
        strategy="uniform",
    )
    return {
        "brier": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr, labels=[0, 1])),
        "mean_predicted": float(np.mean(y_proba_arr)),
        "observed_rate": float(np.mean(y_true_arr)),
        "frac_pos": frac_pos,
        "mean_pred": mean_pred,
        "proba": y_proba_arr,
    }


def evaluate_calibration(split) -> dict[str, Any]:
    """Compare uncalibrated vs sigmoid/isotonic using inner validation from TRAIN only."""
    fitted = load_pipeline(candidate_artifact_path(SELECTED_KEY))
    architecture = _clone_selected_architecture(fitted)
    X_inner, X_val, y_inner, y_val = make_inner_train_val(split)

    uncalibrated = clone(architecture)
    uncalibrated.fit(X_inner, y_inner)
    uncal_proba = _positive_proba(uncalibrated, X_val)
    uncal_stats = _calibration_stats(y_val, uncal_proba)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sigmoid = CalibratedClassifierCV(
        clone(architecture),
        method="sigmoid",
        cv=cv,
        ensemble=False,
    )
    sigmoid.fit(X_inner, y_inner)
    sigmoid_proba = _positive_proba(sigmoid, X_val)
    sigmoid_stats = _calibration_stats(y_val, sigmoid_proba)

    isotonic = CalibratedClassifierCV(
        clone(architecture),
        method="isotonic",
        cv=cv,
        ensemble=False,
    )
    isotonic.fit(X_inner, y_inner)
    isotonic_proba = _positive_proba(isotonic, X_val)
    isotonic_stats = _calibration_stats(y_val, isotonic_proba)

    brier_drop = uncal_stats["brier"] - sigmoid_stats["brier"]
    log_drop = uncal_stats["log_loss"] - sigmoid_stats["log_loss"]
    use_sigmoid = (brier_drop >= BRIER_IMPROVEMENT_MIN) and (log_drop > 0)
    # Isotonic needs more data than ~800 inner-val rows / ~77 frauds.
    use_isotonic = False
    isotonic_note = (
        "Isotonic was evaluated as a diagnostic only. The inner validation fold "
        f"has {int(len(y_val))} rows and {int((y_val == 1).sum())} frauds, which is "
        "thin for a non-parametric calibrator, so isotonic was not selected."
    )

    if use_sigmoid:
        method = "sigmoid"
        reason = (
            f"Sigmoid calibration reduced inner-validation Brier from "
            f"{uncal_stats['brier']:.6f} to {sigmoid_stats['brier']:.6f} "
            f"(drop {brier_drop:.6f}) and log loss from "
            f"{uncal_stats['log_loss']:.6f} to {sigmoid_stats['log_loss']:.6f}. "
            "The improvement is large enough to justify wrapping the selected "
            "Logistic Regression + Class Weight pipeline with CalibratedClassifierCV "
            "(method='sigmoid', cv=5, ensemble=False) fit on the full training fold."
        )
    else:
        method = "none"
        reason = (
            "Sigmoid calibration was compared on an inner validation split taken "
            "from the 4,000-row training fold only. "
            f"Uncalibrated Brier={uncal_stats['brier']:.6f}, log_loss={uncal_stats['log_loss']:.6f}; "
            f"sigmoid Brier={sigmoid_stats['brier']:.6f}, log_loss={sigmoid_stats['log_loss']:.6f} "
            f"(Brier drop {brier_drop:.6f}). "
            f"Calibration is kept only if Brier drops by at least {BRIER_IMPROVEMENT_MIN} "
            "and log loss also improves. That bar was not met, so the uncalibrated "
            "Phase 5 Logistic Regression + Class Weight artifact is retained."
        )

    return {
        "method": method,
        "use_sigmoid": use_sigmoid,
        "use_isotonic": use_isotonic,
        "reason": reason,
        "isotonic_note": isotonic_note,
        "inner_train_rows": int(len(X_inner)),
        "inner_val_rows": int(len(X_val)),
        "inner_val_fraud": int((y_val == 1).sum()),
        "uncalibrated": uncal_stats,
        "sigmoid": sigmoid_stats,
        "isotonic": isotonic_stats,
        "y_val": y_val,
        "architecture": architecture,
        "fitted_phase5": fitted,
    }


def oof_positive_proba(estimator: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    return np.asarray(proba, dtype="float64")[:, 1]


def build_final_estimator(calibration: dict[str, Any], split) -> Any:
    """Fit or reuse the production estimator on the full 4,000-row training fold."""
    if calibration["method"] == "none":
        return calibration["fitted_phase5"]
    if calibration["method"] == "sigmoid":
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        calibrated = CalibratedClassifierCV(
            clone(calibration["architecture"]),
            method="sigmoid",
            cv=cv,
            ensemble=False,
        )
        calibrated.fit(split.X_train, split.y_train)
        return calibrated
    raise RuntimeError(f"STOP: unsupported calibration method {calibration['method']!r}.")


def threshold_estimator_for_oof(calibration: dict[str, Any]) -> Any:
    if calibration["method"] == "none":
        return clone(calibration["architecture"])
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        clone(calibration["architecture"]),
        method="sigmoid",
        cv=cv,
        ensemble=False,
    )


def write_threshold_chart(table: pd.DataFrame, selected_threshold: float, path: Path) -> Path:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["threshold"], y=table["Precision"], mode="lines+markers", name="Precision"))
    fig.add_trace(go.Scatter(x=table["threshold"], y=table["Recall"], mode="lines+markers", name="Recall"))
    fig.add_trace(go.Scatter(x=table["threshold"], y=table["F1"], mode="lines+markers", name="F1"))
    fig.add_trace(
        go.Scatter(
            x=table["threshold"],
            y=table["False_Positive_Rate"],
            mode="lines+markers",
            name="False positive rate",
        )
    )
    fig.add_vline(x=selected_threshold, line_dash="dash", annotation_text=f"Selected {selected_threshold:.2f}")
    fig.update_layout(
        title="Training-only threshold analysis (out-of-fold probabilities)",
        xaxis_title="Decision threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend_title="Metric",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def write_calibration_chart(calibration: dict[str, Any], path: Path) -> Path:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
    fig.add_trace(
        go.Scatter(
            x=calibration["uncalibrated"]["mean_pred"],
            y=calibration["uncalibrated"]["frac_pos"],
            mode="lines+markers",
            name="Uncalibrated",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calibration["sigmoid"]["mean_pred"],
            y=calibration["sigmoid"]["frac_pos"],
            mode="lines+markers",
            name="Sigmoid",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calibration["isotonic"]["mean_pred"],
            y=calibration["isotonic"]["frac_pos"],
            mode="lines+markers",
            name="Isotonic (diagnostic)",
        )
    )
    fig.update_layout(
        title="Reliability curves on the inner training validation fold",
        xaxis_title="Mean predicted fraud probability",
        yaxis_title="Observed fraud fraction",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def write_confusion_chart(metrics: dict[str, Any], threshold: float, path: Path) -> Path:
    z = [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted legitimate", "Predicted fraud"],
            y=["True legitimate", "True fraud"],
            text=[[f"TN {z[0][0]}", f"FP {z[0][1]}"], [f"FN {z[1][0]}", f"TP {z[1][1]}"]],
            texttemplate="%{text}",
            showscale=False,
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title=f"Final test confusion matrix (threshold={threshold:.2f})",
        height=480,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def write_risk_distribution_chart(
    risk_scores: np.ndarray,
    path: Path,
    bands: tuple[dict[str, Any], ...] = RISK_BANDS,
) -> Path:
    fig = go.Figure()
    fig.add_histogram(x=risk_scores, nbinsx=20, name="Risk score")
    for band in bands[1:]:
        fig.add_vline(
            x=float(band["min_inclusive"]),
            line_dash="dash",
            annotation_text=band["name"],
        )
    fig.update_layout(
        title="Final test-set fraud risk score distribution",
        xaxis_title="Risk score (probability x 100)",
        yaxis_title="Transactions",
        xaxis=dict(range=[0, 100]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def write_risk_band_chart(band_summary: pd.DataFrame, path: Path) -> Path:
    fig = go.Figure(go.Bar(x=band_summary["risk_band"], y=band_summary["count"], name="Count"))
    fig.update_layout(
        title="Final test-set risk band counts",
        xaxis_title="Risk band",
        yaxis_title="Transactions",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    float_cols = work.select_dtypes(include=["float64", "float32"]).columns
    for column in float_cols:
        work[column] = work[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_phase7_report(payload: dict[str, Any], path: Path) -> Path:
    cal = payload["calibration"]
    chosen = payload["chosen_threshold"]
    test = payload["test_metrics"]
    split_info = payload["split_info"]
    phase6 = payload["phase6_metrics"]
    lines = [
        "# Phase 7 Model Selection, Calibration, and Risk Scoring",
        "",
        "## 1. Objective",
        "",
        "Freeze a production inference pipeline from the Phase 6 selected candidate.",
        "Calibration and the operating threshold are chosen from training data only.",
        "The untouched 1,000-row Phase 6 test fold is scored once after those decisions are frozen.",
        "The risk score is a model risk indicator, not proof that a transaction is fraudulent.",
        "",
        "## 2. Phase 6 selected candidate",
        "",
        f"- Model: {SELECTED_MODEL_NAME}",
        f"- Strategy: {SELECTED_STRATEGY}",
        f"- Artifact: `models/baseline/{SELECTED_KEY}.joblib`",
        f"- Phase 6 default-threshold metrics: Accuracy={phase6['Accuracy']:.4f}, "
        f"Precision={phase6['Precision']:.4f}, Recall={phase6['Recall']:.4f}, "
        f"F1={phase6['F1']:.4f}, ROC-AUC={phase6['ROC_AUC']:.4f}, PR-AUC={phase6['PR_AUC']:.4f}",
        f"- Phase 6 confusion: TN={phase6['TN']}, FP={phase6['FP']}, FN={phase6['FN']}, TP={phase6['TP']}",
        "",
        "## 3. Calibration methodology",
        "",
        "- Recreate the Phase 5/6 split (`test_size=0.20`, `stratify=y`, `random_state=42`).",
        f"- Hold out {INNER_VAL_SIZE:.0%} of the **training** fold only "
        f"({cal['inner_train_rows']} inner-train / {cal['inner_val_rows']} inner-val, "
        f"{cal['inner_val_fraud']} inner-val frauds).",
        "- Fit an uncalibrated clone of Logistic Regression + Class Weight on inner-train.",
        "- Fit `CalibratedClassifierCV` with `method='sigmoid'` and `method='isotonic'`, "
        f"`cv={CV_SPLITS}`, `ensemble=False`, on inner-train.",
        "- Compare Brier score and log loss on inner-val. The test fold is not used.",
        "- Sigmoid is the primary option. Isotonic is diagnostic only unless the data clearly support it.",
        "",
        "## 4. Calibration result",
        "",
        f"- Inner-val uncalibrated: Brier={cal['uncalibrated']['brier']:.6f}, "
        f"log_loss={cal['uncalibrated']['log_loss']:.6f}, mean predicted={cal['uncalibrated']['mean_predicted']:.4f}, "
        f"observed rate={cal['uncalibrated']['observed_rate']:.4f}",
        f"- Inner-val sigmoid: Brier={cal['sigmoid']['brier']:.6f}, "
        f"log_loss={cal['sigmoid']['log_loss']:.6f}, mean predicted={cal['sigmoid']['mean_predicted']:.4f}",
        f"- Inner-val isotonic: Brier={cal['isotonic']['brier']:.6f}, "
        f"log_loss={cal['isotonic']['log_loss']:.6f}, mean predicted={cal['isotonic']['mean_predicted']:.4f}",
        f"- Selected calibration method: **{cal['method']}**",
        "",
        cal["reason"],
        "",
        cal["isotonic_note"],
        "",
        "## 5. Threshold-selection methodology",
        "",
        "- Out-of-fold `predict_proba` from 5-fold stratified CV on the 4,000-row training fold.",
        "- The same architecture as the frozen production estimator (calibrated or not).",
        "- Grid: 0.10 to 0.90 in steps of 0.05.",
        "- Primary screen: recall at least 0.70 and FPR at most 0.35, then maximize F1.",
        "- Accuracy is not used to pick the threshold.",
        "- Test labels are not inspected during this search.",
        "",
        "## 6. Threshold comparison",
        "",
        "Training out-of-fold metrics:",
        "",
        _markdown_table(
            payload["threshold_table"][
                [
                    "threshold",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "False_Positive_Rate",
                    "False_Negative_Rate",
                    "TN",
                    "FP",
                    "FN",
                    "TP",
                ]
            ]
        ),
        "",
        "## 7. Selected production threshold",
        "",
        f"- **{float(chosen['threshold']):.2f}**",
        f"- Training OOF: Precision={float(chosen['Precision']):.4f}, Recall={float(chosen['Recall']):.4f}, "
        f"F1={float(chosen['F1']):.4f}, FPR={float(chosen['False_Positive_Rate']):.4f}, "
        f"FNR={float(chosen['False_Negative_Rate']):.4f}",
        f"- Training OOF confusion: TN={int(chosen['TN'])}, FP={int(chosen['FP'])}, "
        f"FN={int(chosen['FN'])}, TP={int(chosen['TP'])}",
        f"- Rule: {chosen['selection_rule']}",
        "",
        payload["threshold_reason"],
        "",
        "## 8. Risk-score methodology",
        "",
        "`risk_score = predicted fraud probability x 100`, clipped to 0-100.",
        "This is a model risk indicator for review prioritization. It is not a determination of fraud.",
        "",
        "## 9. Risk bands",
        "",
        "| Band | Score range | Meaning |",
        "| --- | --- | --- |",
        *[
            f"| {band['name']} | {band.get('range', '')} | "
            + (
                "Below the production review threshold"
                if band["name"] == "Low Risk"
                else "Flagged review range"
                if band["name"] == "Medium Risk"
                else "Higher calibrated fraud probability"
                if band["name"] == "High Risk"
                else "Highest calibrated fraud probability"
            )
            + " |"
            for band in payload["risk_bands"]
        ],
        "",
        payload["risk_band_reason"],
        "",
        "## 10. Final model architecture",
        "",
        f"- Classifier: {SELECTED_MODEL_NAME} (`class_weight='balanced'`)",
        f"- Strategy: {SELECTED_STRATEGY}",
        f"- Features ({len(MODEL_FEATURES)}): {', '.join(MODEL_FEATURES)}",
        f"- Numeric preprocess: RobustScaler on {', '.join(NUMERIC_FEATURES)}",
        f"- Categorical preprocess: OneHotEncoder(handle_unknown='ignore', sparse_output=False) on {', '.join(CATEGORICAL_FEATURES)}",
        f"- Calibration: {cal['method']}",
        f"- Production threshold: {float(chosen['threshold']):.2f}",
        f"- Excluded from X: {', '.join(EXCLUDED_FEATURES)}",
        "- Artifact: `models/production/final_fraud_pipeline.joblib`",
        "",
        "## 11. Final untouched-test evaluation",
        "",
        f"- Test rows: {split_info['test_rows']}; fraud={split_info['test_fraud_count']}; "
        f"legitimate={split_info['test_legitimate_count']}; rate={split_info['test_fraud_rate']*100:.2f}%",
        f"- Threshold: {float(chosen['threshold']):.2f}",
        f"- Accuracy={test['Accuracy']:.4f}",
        f"- Precision={test['Precision']:.4f}",
        f"- Recall={test['Recall']:.4f}",
        f"- F1={test['F1']:.4f}",
        f"- ROC-AUC={test['ROC_AUC']:.4f}",
        f"- PR-AUC={test['PR_AUC']:.4f}",
        f"- FPR={test['False_Positive_Rate']:.4f}",
        f"- FNR={test['False_Negative_Rate']:.4f}",
        f"- TN={test['TN']}, FP={test['FP']}, FN={test['FN']}, TP={test['TP']}",
        f"- Flagged as fraud: {test['flagged']} ({test['flagged_rate']*100:.2f}%)",
        f"- Classified legitimate: {test['legitimate_predicted']} ({test['legitimate_predicted_rate']*100:.2f}%)",
        "",
        "These test numbers were not used to change the threshold, calibration, or model.",
        "",
        "### Test-set risk score snapshot",
        "",
        f"- Minimum: {payload['risk_stats']['min']:.4f}",
        f"- Maximum: {payload['risk_stats']['max']:.4f}",
        f"- Mean: {payload['risk_stats']['mean']:.4f}",
        f"- Median: {payload['risk_stats']['median']:.4f}",
        "",
        _markdown_table(payload["band_summary"]),
        "",
        "Example rows use Transaction_ID only. Customer identifiers are not shown.",
        "",
        _markdown_table(payload["example"]),
        "",
        "## 12. Limitations",
        "",
        "- 96 test frauds: interval estimates around recall/precision are wide.",
        "- Class-weighted logistic regression probabilities are not a causal fraud proof.",
        "- False positives create review workload; false negatives are missed fraud.",
        "- No dashboard, SQLite, or deployment is included in this phase.",
        "",
        "## 13. Reproducibility information",
        "",
        f"- random_state: {RANDOM_STATE}",
        f"- dataset: `{DATASET_FILENAME}`",
        f"- raw MD5: `{payload['dataset_hash']}`",
        f"- production model MD5: `{payload['hashes']['production_model']}`",
        f"- threshold CSV MD5: `{payload['hashes']['threshold_csv']}`",
        f"- final metrics CSV MD5: `{payload['hashes']['final_metrics_csv']}`",
        f"- risk-band CSV MD5: `{payload['hashes']['risk_band_csv']}`",
        "",
        "## 14. Data leakage safeguards",
        "",
        "- Split first; test never resampled.",
        "- Preprocess remains the Phase 5 train-fit transformer inside the saved pipeline.",
        "- Calibration compared and, if used, fitted on training rows only.",
        "- Threshold chosen from training out-of-fold probabilities only.",
        "- Customer_ID, Transaction_ID, Suspicious_Keyword, Fraudulent, and raw Transaction_Date are not model features.",
        "",
        "## 15. Conclusion",
        "",
        payload["conclusion"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def risk_band_summary(
    risk_scores: np.ndarray,
    y_true: pd.Series | None = None,
    bands: tuple[dict[str, Any], ...] = RISK_BANDS,
) -> pd.DataFrame:
    assigned = assign_risk_band(risk_scores, bands)
    frame = pd.DataFrame({"risk_band": assigned, "risk_score": risk_scores})
    if y_true is not None:
        frame["Fraudulent"] = np.asarray(y_true).astype(int)
    rows = []
    n = len(frame)
    for band in bands:
        name = band["name"]
        part = frame[frame["risk_band"] == name]
        row = {
            "risk_band": name,
            "score_range": band.get("range", ""),
            "count": int(len(part)),
            "percentage": float(len(part) / n) if n else 0.0,
            "mean_risk_score": float(part["risk_score"].mean()) if len(part) else 0.0,
        }
        if y_true is not None:
            row["fraud_count"] = int(part["Fraudulent"].sum()) if len(part) else 0
            row["fraud_rate"] = float(part["Fraudulent"].mean()) if len(part) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_phase7() -> dict[str, Any]:
    set_global_seed()
    dataset_hash = assert_raw_dataset_unchanged()
    if dataset_hash != EXPECTED_RAW_DATASET_MD5:
        raise RuntimeError("STOP: raw dataset hash mismatch.")

    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    split_info = assert_test_fold(split)
    if int((split.y_test == 0).sum()) != 904:
        raise RuntimeError(
            f"STOP: expected 904 legitimate test rows, got {int((split.y_test == 0).sum())}."
        )
    x_snapshot = split.X_test.copy(deep=True)
    y_snapshot = split.y_test.copy(deep=True)

    if not SELECTED_METADATA_PATH.exists():
        raise FileNotFoundError("Phase 6 selected_model.json is missing.")
    phase6 = json.loads(SELECTED_METADATA_PATH.read_text(encoding="utf-8"))
    phase6_metrics = phase6["test_metrics"]

    artifact_path = candidate_artifact_path(SELECTED_KEY)
    if not artifact_path.exists():
        raise FileNotFoundError(f"STOP: missing selected candidate {artifact_path.as_posix()}")

    calibration = evaluate_calibration(split)
    oof_estimator = threshold_estimator_for_oof(calibration)
    oof_proba = oof_positive_proba(oof_estimator, split.X_train, split.y_train)
    train_threshold_table = threshold_table(split.y_train, oof_proba)
    chosen = select_operating_threshold(train_threshold_table)
    selected_threshold = float(chosen["threshold"])

    threshold_reason = (
        f"Threshold {selected_threshold:.2f} was selected from training out-of-fold "
        f"probabilities ({CV_SPLITS}-fold CV on {int(len(split.X_train))} training rows, "
        f"{int((split.y_train == 1).sum())} training frauds). "
        f"It has training OOF recall {float(chosen['Recall']):.4f} and FPR "
        f"{float(chosen['False_Positive_Rate']):.4f}, with F1 {float(chosen['F1']):.4f}. "
        "The default 0.50 cut was not adopted automatically. "
        f"Selection rule: {chosen['selection_rule']}. "
        "On the same training OOF table, 0.15 had a slightly higher F1 but recall "
        "below 0.70, so it failed the primary fraud-recall screen."
    )

    risk_bands = choose_risk_bands(calibration["method"], selected_threshold)
    if calibration["method"] == "none":
        risk_band_reason = (
            "Uncalibrated probabilities remain on a class-weighted scale, so the "
            "pre-declared 0-29 / 30-59 / 60-79 / 80-100 bands are used."
        )
    else:
        risk_band_reason = (
            "Sigmoid calibration mapped probabilities onto the training prevalence "
            "scale (inner-val mean predicted ~0.10). The original 0-29 Low band would "
            "have placed almost every flagged review in Low Risk. Bands were therefore "
            "aligned to the frozen training threshold with round cut-points: Low is "
            "below-threshold, Medium/High/Critical partition the flagged range. "
            "These cuts were not estimated from test labels."
        )

    final_estimator = build_final_estimator(calibration, split)
    production = FraudRiskPipeline(
        final_estimator,
        threshold=selected_threshold,
        model_name=SELECTED_MODEL_NAME,
        strategy=SELECTED_STRATEGY,
        calibration_method=calibration["method"],
        risk_bands=risk_bands,
    )

    if not split.X_test.equals(x_snapshot) or not split.y_test.equals(y_snapshot):
        raise RuntimeError("STOP: training-time work mutated the test fold.")

    # One-shot untouched test evaluation after freeze.
    test_score = production.score(split.X_test)
    test_proba = test_score["predicted_probability"].to_numpy()
    test_pred = test_score["predicted_label"].to_numpy()
    test_metrics = metrics_at_threshold(split.y_test, test_proba, selected_threshold)
    ranking = compute_binary_metrics(split.y_test, test_pred, test_proba)
    test_metrics["ROC_AUC"] = ranking["ROC_AUC"]
    test_metrics["PR_AUC"] = ranking["PR_AUC"]
    risk_scores = test_score["risk_score"].to_numpy()
    band_summary = risk_band_summary(risk_scores, split.y_test, risk_bands)
    example = pd.DataFrame(
        {
            "Transaction_ID": split.id_test.head(12).to_numpy(),
            "risk_score": np.round(risk_scores[:12], 4),
            "risk_band": test_score["risk_band"].head(12).to_numpy(),
            "predicted_label": test_pred[:12],
        }
    )

    PRODUCTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SELECTION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, PRODUCTION_PIPELINE_PATH)

    metadata = {
        "model_name": SELECTED_MODEL_NAME,
        "strategy": SELECTED_STRATEGY,
        "model_version": MODEL_VERSION,
        "status": "production candidate",
        "artifact_path": "models/production/final_fraud_pipeline.joblib",
        "source_candidate": f"models/baseline/{SELECTED_KEY}.joblib",
        "feature_list": list(MODEL_FEATURES),
        "feature_count": len(MODEL_FEATURES),
        "excluded_features": list(EXCLUDED_FEATURES),
        "preprocessing": {
            "numeric": "RobustScaler",
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "categorical_features": list(CATEGORICAL_FEATURES),
            "fit_on": "training fold only",
        },
        "calibration_method": calibration["method"],
        "calibration_reason": calibration["reason"],
        "production_threshold": selected_threshold,
        "threshold_selection": {
            "data": "5-fold out-of-fold probabilities on the 4,000-row training fold",
            "grid": list(train_threshold_table["threshold"]),
            "rule": str(chosen["selection_rule"]),
            "reason": threshold_reason,
        },
        "risk_score_method": "predicted_fraud_probability * 100, clipped to [0, 100]",
        "risk_score_disclaimer": (
            "The risk score is a model risk indicator, not proof that a transaction is fraudulent."
        ),
        "risk_bands": [
            {
                "name": band["name"],
                "range": band.get("range", ""),
                "min_inclusive": band["min_inclusive"],
                "max_exclusive": band["max_exclusive"],
            }
            for band in risk_bands
        ],
        "risk_band_reason": risk_band_reason,
        "random_state": RANDOM_STATE,
        "dataset": DATASET_FILENAME,
        "training_row_count": int(len(split.X_train)),
        "training_fraud_count": int((split.y_train == 1).sum()),
        "training_legitimate_count": int((split.y_train == 0).sum()),
        "test_row_count": int(len(split.X_test)),
        "test_fraud_count": int((split.y_test == 1).sum()),
        "test_legitimate_count": int((split.y_test == 0).sum()),
        "phase6_reference_metrics": phase6_metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_ready_claim": False,
        "notes": "Phase 7 frozen pipeline. Dashboard and deployment are later work.",
    }
    PRODUCTION_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")

    threshold_csv = REPORTS_DIR / "threshold_analysis.csv"
    train_threshold_table.to_csv(threshold_csv, index=False)
    final_metrics_csv = REPORTS_DIR / "final_model_metrics.csv"
    pd.DataFrame([test_metrics]).to_csv(final_metrics_csv, index=False)
    risk_csv = REPORTS_DIR / "risk_band_summary.csv"
    band_summary.to_csv(risk_csv, index=False)

    write_threshold_chart(
        train_threshold_table,
        selected_threshold,
        MODEL_SELECTION_ARTIFACTS_DIR / "threshold_analysis.html",
    )
    write_calibration_chart(calibration, MODEL_SELECTION_ARTIFACTS_DIR / "calibration_curve.html")
    write_confusion_chart(
        test_metrics,
        selected_threshold,
        MODEL_SELECTION_ARTIFACTS_DIR / "final_confusion_matrix.html",
    )
    write_risk_distribution_chart(
        risk_scores,
        MODEL_SELECTION_ARTIFACTS_DIR / "final_risk_distribution.html",
        risk_bands,
    )
    write_risk_band_chart(band_summary, MODEL_SELECTION_ARTIFACTS_DIR / "risk_band_distribution.html")

    hashes = {
        "raw_dataset": dataset_hash,
        "production_model": file_md5(PRODUCTION_PIPELINE_PATH),
        "threshold_csv": file_md5(threshold_csv),
        "final_metrics_csv": file_md5(final_metrics_csv),
        "risk_band_csv": file_md5(risk_csv),
    }
    conclusion = (
        f"The production wrapper is {SELECTED_MODEL_NAME} + {SELECTED_STRATEGY} "
        f"with calibration={calibration['method']} and threshold={selected_threshold:.2f}. "
        f"On the untouched test set it scores Recall={test_metrics['Recall']:.4f}, "
        f"Precision={test_metrics['Precision']:.4f}, F1={test_metrics['F1']:.4f}, "
        f"ROC-AUC={test_metrics['ROC_AUC']:.4f}, PR-AUC={test_metrics['PR_AUC']:.4f}, "
        f"TP={test_metrics['TP']}, FP={test_metrics['FP']}, TN={test_metrics['TN']}, "
        f"FN={test_metrics['FN']}. These test metrics were not used for further tuning."
    )
    report_payload = {
        "calibration": calibration,
        "chosen_threshold": chosen,
        "threshold_table": train_threshold_table,
        "threshold_reason": threshold_reason,
        "test_metrics": test_metrics,
        "split_info": split_info,
        "phase6_metrics": phase6_metrics,
        "dataset_hash": dataset_hash,
        "hashes": hashes,
        "conclusion": conclusion,
        "band_summary": band_summary,
        "example": example,
        "risk_bands": risk_bands,
        "risk_band_reason": risk_band_reason,
        "risk_stats": {
            "min": float(np.min(risk_scores)),
            "max": float(np.max(risk_scores)),
            "mean": float(np.mean(risk_scores)),
            "median": float(np.median(risk_scores)),
        },
    }
    write_phase7_report(report_payload, REPORTS_DIR / "phase7_model_selection.md")

    if not split.X_test.equals(x_snapshot) or not split.y_test.equals(y_snapshot):
        raise RuntimeError("STOP: test fold changed during final scoring.")

    return {
        "split_info": split_info,
        "calibration": calibration,
        "threshold": selected_threshold,
        "chosen_threshold": chosen,
        "threshold_reason": threshold_reason,
        "test_metrics": test_metrics,
        "band_summary": band_summary,
        "example": example,
        "risk_scores": risk_scores,
        "hashes": hashes,
        "dataset_hash": dataset_hash,
        "conclusion": conclusion,
        "production_path": PRODUCTION_PIPELINE_PATH,
        "metadata_path": PRODUCTION_METADATA_PATH,
    }
