"""Fair comparison of Phase 5 candidate pipelines on the untouched test set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.eda import raw_dataset_md5
from src.evaluation.metrics import (
    compute_binary_metrics,
    dummy_always_legitimate,
    pr_points,
    roc_points,
)
from src.features.feature_engineering import MODEL_FEATURES
from src.models.pipeline import load_xy, make_stratified_split
from src.utils.paths import (
    BASELINE_MODELS_DIR,
    DATASET_FILENAME,
    EVALUATION_ARTIFACTS_DIR,
    REPORTS_DIR,
)
from src.utils.seeds import RANDOM_STATE

EXPECTED_RAW_DATASET_MD5 = "9a4a90ce2e07a717b4289dc95a71663c"
EXPECTED_TEST_ROWS = 1000
EXPECTED_TEST_FRAUD_COUNT = 96
EXPECTED_TEST_FRAUD_RATE_PCT = 9.60
MAX_PRACTICAL_FALSE_POSITIVES = 300

CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("logistic_regression_smote", "Logistic Regression", "SMOTE"),
    ("decision_tree_smote", "Decision Tree", "SMOTE"),
    ("random_forest_smote", "Random Forest", "SMOTE"),
    ("xgboost_smote", "XGBoost", "SMOTE"),
    ("logistic_regression_class_weight", "Logistic Regression", "Class Weight"),
    ("decision_tree_class_weight", "Decision Tree", "Class Weight"),
    ("random_forest_class_weight", "Random Forest", "Class Weight"),
    ("xgboost_class_weight", "XGBoost", "Class Weight"),
)

FAMILY_COMPLEXITY = {
    "Logistic Regression": 0,
    "Decision Tree": 1,
    "Random Forest": 2,
    "XGBoost": 3,
}
STRATEGY_COMPLEXITY = {"Class Weight": 0, "SMOTE": 1}

DISPLAY_COLS = [
    "Model",
    "Strategy",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "TN",
    "FP",
    "FN",
    "TP",
]


def candidate_artifact_path(key: str) -> Path:
    return BASELINE_MODELS_DIR / f"{key}.joblib"


def assert_raw_dataset_unchanged() -> str:
    """STOP if the primary CSV bytes changed since Phase 1."""
    digest = raw_dataset_md5()
    if digest != EXPECTED_RAW_DATASET_MD5:
        raise RuntimeError(
            "STOP: raw dataset hash changed. "
            f"expected {EXPECTED_RAW_DATASET_MD5}, got {digest}. "
            "Phase 6 will not evaluate on a mutated file."
        )
    return digest


def assert_test_fold(split: Any) -> dict[str, Any]:
    """Verify the recreated test fold matches the locked Phase 5 split."""
    test_rows = int(len(split.X_test))
    test_fraud_count = int((split.y_test == 1).sum())
    test_fraud_rate = float((split.y_test == 1).mean())
    test_fraud_rate_pct = round(test_fraud_rate * 100, 2)
    if test_rows != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"STOP: expected {EXPECTED_TEST_ROWS} test rows, got {test_rows}."
        )
    if test_fraud_count != EXPECTED_TEST_FRAUD_COUNT:
        raise RuntimeError(
            f"STOP: expected {EXPECTED_TEST_FRAUD_COUNT} test frauds, got {test_fraud_count}."
        )
    if test_fraud_rate_pct != EXPECTED_TEST_FRAUD_RATE_PCT:
        raise RuntimeError(
            f"STOP: expected test fraud rate {EXPECTED_TEST_FRAUD_RATE_PCT:.2f}%, "
            f"got {test_fraud_rate_pct:.2f}%."
        )
    return {
        "test_rows": test_rows,
        "test_fraud_count": test_fraud_count,
        "test_legitimate_count": int((split.y_test == 0).sum()),
        "test_fraud_rate": test_fraud_rate,
        "test_fraud_rate_pct": test_fraud_rate_pct,
        "train_rows": int(len(split.X_train)),
        "train_fraud_count": int((split.y_train == 1).sum()),
        "feature_count": int(split.X_test.shape[1]),
    }


def load_candidates() -> dict[str, Any]:
    loaded = {}
    missing = []
    broken = []
    for key, _name, _strategy in CANDIDATES:
        path = candidate_artifact_path(key)
        if not path.exists():
            missing.append(path.as_posix())
            continue
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "predict_proba"):
            broken.append(f"{key}: missing predict_proba()")
        loaded[key] = pipeline
    if missing:
        raise FileNotFoundError("Missing Phase 5 artifacts:\n" + "\n".join(missing))
    if broken:
        raise RuntimeError("STOP: Phase 5 artifact problem:\n" + "\n".join(broken))
    if len(loaded) != len(CANDIDATES):
        raise RuntimeError("STOP: expected eight candidate pipelines.")
    return loaded


def load_evaluation_split():
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    return split


def _predict(pipeline: Any, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(pipeline.predict(X_test)).astype(int)
    proba = np.asarray(pipeline.predict_proba(X_test), dtype="float64")
    if proba.shape != (len(X_test), 2):
        raise ValueError(f"predict_proba shape {proba.shape} is not (n, 2).")
    y_proba = proba[:, 1]
    return y_pred, y_proba


def evaluate_candidates(split=None, pipelines: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Score every candidate on the same X_test / y_test. Does not refit."""
    split = split or load_evaluation_split()
    pipelines = pipelines or load_candidates()
    y_test = split.y_test
    X_test = split.X_test
    rows = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for key, model_name, strategy in CANDIDATES:
        y_pred, y_proba = _predict(pipelines[key], X_test)
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        rows.append(
            {
                "key": key,
                "Model": model_name,
                "Strategy": strategy,
                **metrics,
            }
        )
        fpr, tpr, roc_thresholds = roc_points(y_test, y_proba)
        precision, recall, pr_thresholds = pr_points(y_test, y_proba)
        curves[key] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
        }
    dummy = dummy_always_legitimate(y_test)
    rows.append(
        {
            "key": "dummy_always_legitimate",
            "Model": "Always Legitimate",
            "Strategy": "Dummy",
            **dummy,
        }
    )
    comparison = pd.DataFrame(rows)
    comparison["confusion_total"] = (
        comparison["TN"] + comparison["FP"] + comparison["FN"] + comparison["TP"]
    )
    return comparison, curves


def select_candidate(comparison: pd.DataFrame) -> pd.Series:
    """Pick a selected candidate from computed metrics. Not a production model.

    Ranking: F1 first (precision/recall balance), then PR-AUC, recall, precision,
    ROC-AUC. If F1 is within 0.01 of the leader, prefer the simpler family, then
    class-weight over SMOTE. Models with FP > 300 (30% of the 1,000-row test set)
    are treated as operationally heavy and lose the F1 tie-break.
    """
    ml = comparison[comparison["Strategy"] != "Dummy"].copy()
    ml["family_complexity"] = ml["Model"].map(FAMILY_COMPLEXITY)
    ml["strategy_complexity"] = ml["Strategy"].map(STRATEGY_COMPLEXITY)
    ml["impractical_fp"] = (ml["FP"] > MAX_PRACTICAL_FALSE_POSITIVES).astype(int)
    ranked = ml.sort_values(
        by=[
            "impractical_fp",
            "F1",
            "PR_AUC",
            "Recall",
            "Precision",
            "ROC_AUC",
            "family_complexity",
            "strategy_complexity",
        ],
        ascending=[True, False, False, False, False, False, True, True],
    )
    top = ranked.iloc[0]
    close = ranked[
        (ranked["impractical_fp"] == top["impractical_fp"])
        & (ranked["F1"] >= float(top["F1"]) - 0.01)
    ]
    close = close.sort_values(
        by=["F1", "PR_AUC", "Recall", "Precision", "ROC_AUC", "family_complexity", "strategy_complexity"],
        ascending=[False, False, False, False, False, True, True],
    )
    return close.iloc[0]


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_comparison_csv(comparison: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (REPORTS_DIR / "model_comparison.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    trained = comparison[comparison["Strategy"] != "Dummy"][DISPLAY_COLS]
    trained.to_csv(path, index=False)
    return path


def write_selected_model_json(selected: pd.Series, path: Path | None = None) -> Path:
    path = path or (BASELINE_MODELS_DIR / "selected_model.json")
    payload = {
        "status": "selected candidate",
        "model_name": selected["Model"],
        "strategy": selected["Strategy"],
        "artifact_key": selected["key"],
        "artifact_path": f"models/baseline/{selected['key']}.joblib",
        "selection_reason": selected.get(
            "selection_reason",
            "Highest F1 among candidates, with PR-AUC/recall/ROC-AUC and complexity used as tie-breakers. Default threshold only.",
        ),
        "test_metrics": {
            "Accuracy": float(selected["Accuracy"]),
            "Precision": float(selected["Precision"]),
            "Recall": float(selected["Recall"]),
            "F1": float(selected["F1"]),
            "ROC_AUC": float(selected["ROC_AUC"]),
            "PR_AUC": float(selected["PR_AUC"]),
            "TN": int(selected["TN"]),
            "FP": int(selected["FP"]),
            "FN": int(selected["FN"]),
            "TP": int(selected["TP"]),
        },
        "random_state": RANDOM_STATE,
        "dataset": DATASET_FILENAME,
        "feature_count": len(MODEL_FEATURES),
        "threshold": "default (0.5 / model default). Production threshold is Phase 7.",
        "production_ready": False,
        "role": "selected candidate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return path


def write_comparison_markdown(
    comparison: pd.DataFrame,
    selected: pd.Series,
    split_info: dict[str, Any],
    path: Path | None = None,
) -> Path:
    path = path or (REPORTS_DIR / "model_comparison.md")
    ml = comparison[comparison["Strategy"] != "Dummy"].copy()
    dummy = comparison[comparison["Strategy"] == "Dummy"].iloc[0]
    selected_row = ml[(ml["key"] == selected["key"])].iloc[0]
    lines = [
        "# Phase 6 Model Comparison",
        "",
        "All figures below were computed by scoring saved Phase 5 pipelines on one untouched test set.",
        "No hyperparameters were changed. The default decision threshold was used. This is a **selected candidate**, not a production model.",
        "",
        "## 1. Evaluation methodology",
        "",
        "- Load `data/raw/financial_fraud_detection_dataset.csv`",
        "- `build_features()` from Phase 4 (no target, IDs, or Suspicious_Keyword in X)",
        "- Recreate the Phase 5 split: `test_size=0.20`, `stratify=y`, `random_state=42`",
        "- Load eight `joblib` pipelines; do not refit",
        "- `y_pred = predict(X_test)`; `y_proba = predict_proba(X_test)[:, 1]`",
        "- ROC-AUC and PR-AUC use probabilities, not hard labels",
        "- Precision / recall / F1 use `average='binary'` and `zero_division=0`",
        "",
        "## 2. Test set",
        "",
        f"- Rows: {split_info['test_rows']}",
        f"- Fraud count: {split_info['test_fraud_count']}",
        f"- Legitimate count: {split_info['test_legitimate_count']}",
        f"- Fraud rate: {_pct(split_info['test_fraud_rate'])}",
        f"- Train rows (not used for scoring): {split_info['train_rows']}",
        "",
        f"A classifier that predicts every row as legitimate would already be about {_pct(dummy['Accuracy'])} accurate while catching **zero** fraud. Accuracy alone is therefore not the selection metric.",
        "",
        "## 3. Model comparison table",
        "",
        _markdown_table(ml[DISPLAY_COLS]),
        "",
        "### Dummy baseline (not a trained model)",
        "",
        "Always Legitimate predicts every test row as class 0. It is included only to show why accuracy is a poor fraud metric.",
        "",
        _markdown_table(comparison[comparison["Strategy"] == "Dummy"][DISPLAY_COLS]),
        "",
        f"Constant-zero scores yield ROC-AUC={_num(dummy['ROC_AUC'])} (no ranking) and PR-AUC={_num(dummy['PR_AUC'])} (equal to prevalence when scores are constant). These AUC values are diagnostic, not competitive.",
        "",
        "## 4. Confusion matrix summary",
        "",
        "Counts are from the 1,000-row test set (TN+FP+FN+TP = 1,000 for every row).",
        "",
        _markdown_table(ml[["Model", "Strategy", "TN", "FP", "FN", "TP"]]),
        "",
        "## 5. ROC-AUC comparison",
        "",
        _markdown_table(ml.sort_values("ROC_AUC", ascending=False)[["Model", "Strategy", "ROC_AUC"]]),
        "",
        "## 6. PR-AUC comparison",
        "",
        "PR-AUC is the more informative ranking metric at 9.6% prevalence.",
        "",
        _markdown_table(ml.sort_values("PR_AUC", ascending=False)[["Model", "Strategy", "PR_AUC"]]),
        "",
        "## 7. Precision / recall tradeoff",
        "",
        _markdown_table(
            ml.sort_values("F1", ascending=False)[["Model", "Strategy", "Precision", "Recall", "F1", "FP", "FN"]]
        ),
        "",
        "## 8. False-positive analysis",
        "",
        f"A false positive is a legitimate test transaction flagged as fraud. The test set has {split_info['test_legitimate_count']} legitimate rows.",
        "",
        *[
            f"- {row.Model} ({row.Strategy}): FP={int(row.FP)} (precision={_num(row.Precision)})"
            for row in ml.sort_values("FP").itertuples()
        ],
        "",
        "## 9. False-negative analysis",
        "",
        f"A false negative is a fraudulent test transaction the model missed. The test set has {split_info['test_fraud_count']} fraud rows.",
        "",
        *[
            f"- {row.Model} ({row.Strategy}): FN={int(row.FN)}, TP={int(row.TP)} (recall={_num(row.Recall)})"
            for row in ml.sort_values("FN").itertuples()
        ],
        "",
        "## 10. Recommended model",
        "",
        f"**Selected candidate:** {selected_row['Model']} — {selected_row['Strategy']}",
        "",
        f"- Artifact: `models/baseline/{selected_row['key']}.joblib`",
        f"- Accuracy={_num(selected_row['Accuracy'])}, Precision={_num(selected_row['Precision'])}, Recall={_num(selected_row['Recall'])}, F1={_num(selected_row['F1'])}",
        f"- ROC-AUC={_num(selected_row['ROC_AUC'])}, PR-AUC={_num(selected_row['PR_AUC'])}",
        f"- TN={int(selected_row['TN'])}, FP={int(selected_row['FP'])}, FN={int(selected_row['FN'])}, TP={int(selected_row['TP'])}",
        "",
        f"On this test set the selected candidate caught {int(selected_row['TP'])} of {split_info['test_fraud_count']} frauds (true positives) and missed {int(selected_row['FN'])} frauds (false negatives: fraudulent transactions the model labeled legitimate). It correctly cleared {int(selected_row['TN'])} legitimate transactions and incorrectly flagged {int(selected_row['FP'])} legitimate transactions as fraud (false positives). Those {int(selected_row['FP'])} false positives would create review workload; the {int(selected_row['FN'])} false negatives are missed fraud loss.",
        "",
        "## 11. Reason for recommendation",
        "",
        selected.get(
            "selection_reason",
            "See selected_model.json.",
        ),
        "",
        "Dummy Always Legitimate is shown only to illustrate the accuracy trap. It is not a trained rival.",
        "",
        "## 12. Limitations",
        "",
        "- Test size is 1,000 rows / 96 fraud cases, so metric variance is non-trivial.",
        "- Default threshold only; Phase 7 may change operating point.",
        "- No probability calibration yet.",
        "- SMOTE vs class-weight is a training choice already locked in Phase 5 artifacts.",
        "- This candidate is not production-ready.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        if column in work.columns:
            work[column] = work[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_evaluation_charts(
    comparison: pd.DataFrame,
    curves: dict[str, dict[str, np.ndarray]],
    y_test: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or EVALUATION_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ml = comparison[comparison["Strategy"] != "Dummy"].copy()
    ml["label"] = ml["Model"] + " / " + ml["Strategy"]

    metric_fig = go.Figure()
    for metric in ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        metric_fig.add_bar(x=ml["label"], y=ml[metric], name=metric)
    metric_fig.update_layout(
        title="Test-set metric comparison (default threshold)",
        xaxis_title="Model / strategy",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        legend_title="Metric",
    )
    metric_path = output_dir / "model_metric_comparison.html"
    metric_fig.write_html(metric_path, include_plotlyjs="cdn", full_html=True)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    for key, model_name, strategy in CANDIDATES:
        roc_fig.add_trace(
            go.Scatter(
                x=curves[key]["fpr"],
                y=curves[key]["tpr"],
                mode="lines",
                name=f"{model_name} / {strategy}",
            )
        )
    roc_fig.update_layout(
        title="ROC curves on the untouched test set",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    roc_path = output_dir / "roc_curves.html"
    roc_fig.write_html(roc_path, include_plotlyjs="cdn", full_html=True)

    prevalence = float((np.asarray(y_test) == 1).mean())
    pr_fig = go.Figure()
    pr_fig.add_hline(y=prevalence, line_dash="dash", annotation_text=f"Prevalence {prevalence:.4f}")
    for key, model_name, strategy in CANDIDATES:
        pr_fig.add_trace(
            go.Scatter(
                x=curves[key]["recall"],
                y=curves[key]["precision"],
                mode="lines",
                name=f"{model_name} / {strategy}",
            )
        )
    pr_fig.update_layout(
        title="Precision-recall curves on the untouched test set",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    pr_path = output_dir / "precision_recall_curves.html"
    pr_fig.write_html(pr_path, include_plotlyjs="cdn", full_html=True)

    fig_cm = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[f"{name} / {strategy}" for _key, name, strategy in CANDIDATES],
    )
    for index, (key, _name, _strategy) in enumerate(CANDIDATES):
        row = comparison[comparison["key"] == key].iloc[0]
        z = [[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]]
        r, c = divmod(index, 4)
        fig_cm.add_trace(
            go.Heatmap(
                z=z,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                text=[[f"TN {z[0][0]}", f"FP {z[0][1]}"], [f"FN {z[1][0]}", f"TP {z[1][1]}"]],
                texttemplate="%{text}",
                showscale=False,
                colorscale="Blues",
            ),
            row=r + 1,
            col=c + 1,
        )
    fig_cm.update_layout(title="Confusion matrices on the 1,000-row test set", height=640)
    cm_path = output_dir / "confusion_matrices.html"
    fig_cm.write_html(cm_path, include_plotlyjs="cdn", full_html=True)

    return {
        "model_metric_comparison": metric_path,
        "roc_curves": roc_path,
        "precision_recall_curves": pr_path,
        "confusion_matrices": cm_path,
    }


def build_selection_reason(comparison: pd.DataFrame, selected: pd.Series) -> str:
    ml = comparison[comparison["Strategy"] != "Dummy"]
    best_recall = ml.loc[ml["Recall"].idxmax()]
    best_pr = ml.loc[ml["PR_AUC"].idxmax()]
    dummy = comparison[comparison["Strategy"] == "Dummy"].iloc[0]
    parts = [
        f"{selected['Model']} ({selected['Strategy']}) is the selected candidate because it has the strongest practical F1 "
        f"({float(selected['F1']):.4f}) at the default threshold, with precision {float(selected['Precision']):.4f} "
        f"and recall {float(selected['Recall']):.4f}.",
        f"On the 1,000-row test set it yields TP={int(selected['TP'])}, FN={int(selected['FN'])}, "
        f"FP={int(selected['FP'])}, TN={int(selected['TN'])}.",
        f"PR-AUC={float(selected['PR_AUC']):.4f} and ROC-AUC={float(selected['ROC_AUC']):.4f}.",
        f"The Always Legitimate dummy reaches accuracy {float(dummy['Accuracy']):.4f} with recall 0, which is why accuracy was not used to pick a winner.",
    ]
    if best_recall["key"] != selected["key"]:
        parts.append(
            f"The highest-recall model is {best_recall['Model']} ({best_recall['Strategy']}) "
            f"at recall {float(best_recall['Recall']):.4f} with FP={int(best_recall['FP'])} and precision {float(best_recall['Precision']):.4f}; "
            "that extra recall was not chosen if it inflated false positives without a better F1/PR-AUC balance."
        )
    if best_pr["key"] != selected["key"]:
        parts.append(
            f"Highest PR-AUC is {best_pr['Model']} ({best_pr['Strategy']}) at {float(best_pr['PR_AUC']):.4f}."
        )
    parts.append("If two models were within 0.01 F1, the simpler family / class-weight pipeline was preferred. Threshold calibration is deferred to Phase 7.")
    return " ".join(parts)


def run_evaluation() -> dict[str, Any]:
    dataset_hash = assert_raw_dataset_unchanged()
    split = load_evaluation_split()
    split_info = assert_test_fold(split)
    split_info["dataset_hash"] = dataset_hash
    pipelines = load_candidates()
    x_snapshot = split.X_test.copy(deep=True)
    y_snapshot = split.y_test.copy(deep=True)
    comparison, curves = evaluate_candidates(split, pipelines)
    if not split.X_test.equals(x_snapshot) or not split.y_test.equals(y_snapshot):
        raise RuntimeError("STOP: scoring mutated the test set.")
    totals = comparison["confusion_total"]
    if not (totals == EXPECTED_TEST_ROWS).all():
        raise RuntimeError("STOP: a confusion matrix does not sum to 1,000.")
    selected = select_candidate(comparison)
    reason = build_selection_reason(comparison, selected)
    selected = selected.copy()
    selected["selection_reason"] = reason
    comparison_rows = comparison.copy()
    csv_path = write_comparison_csv(comparison_rows)
    json_path = write_selected_model_json(selected)
    md_path = write_comparison_markdown(comparison_rows, selected, split_info)
    charts = write_evaluation_charts(comparison_rows, curves, split.y_test)
    return {
        "comparison": comparison_rows,
        "selected": selected,
        "split_info": split_info,
        "csv_path": csv_path,
        "md_path": md_path,
        "json_path": json_path,
        "charts": charts,
        "X_test": split.X_test,
        "y_test": split.y_test,
        "curves": curves,
        "pipelines": pipelines,
        "dataset_hash": dataset_hash,
    }
