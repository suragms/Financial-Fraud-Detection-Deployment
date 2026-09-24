"""Leakage-safe machine-learning training pipelines.

Workflow:
    raw CSV
    -> Phase 4 build_features()
    -> stratified train/test split
    -> fit preprocessor on train only
    -> optional SMOTE on transformed train only
    -> fit model on train
    -> untouched test set

This phase trains candidate baselines. It does not select a winner, tune
thresholds on the test set, or build the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


from src.data.loader import TARGET_COLUMN, load_fraud_dataset
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    extract_metadata,
    extract_target,
)
from src.utils.paths import BASELINE_MODELS_DIR, DATASET_FILENAME, PROJECT_ROOT, REPORTS_DIR
from src.utils.seeds import RANDOM_STATE, set_global_seed

TEST_SIZE = 0.20
SMOTE_K_NEIGHBORS = 5


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    id_train: pd.Series
    id_test: pd.Series


def load_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data and return X, y, and Transaction_ID metadata."""
    raw = load_fraud_dataset()
    features = build_features(raw)
    target = extract_target(raw)
    metadata = extract_metadata(raw)["Transaction_ID"]
    if TARGET_COLUMN in features.columns:
        raise ValueError("Target leaked into the feature matrix.")
    if list(features.columns) != list(MODEL_FEATURES):
        raise ValueError("Feature columns do not match MODEL_FEATURES.")
    return features, target, metadata


def make_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """One stratified split. The test fold is never resampled."""
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        id_train=id_train.reset_index(drop=True),
        id_test=id_test.reset_index(drop=True),
    )


def build_preprocessor() -> ColumnTransformer:
    """RobustScaler + OneHotEncoder. Fit this on training data only.

    OneHotEncoder uses dense output because the table is small (5,000 rows)
    and SMOTE cannot run on a sparse matrix. handle_unknown='ignore' keeps
    future Streamlit inputs from crashing on unseen categories.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", RobustScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_smote_pipeline(estimator: Any) -> ImbPipeline:
    """Preprocess -> SMOTE (train/fit only) -> classifier."""
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
            ("model", estimator),
        ]
    )


def build_class_weight_pipeline(estimator: Any) -> ImbPipeline:
    """Preprocess -> classifier. No SMOTE. Class weighting lives on the estimator."""
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _logistic(*, class_weight: str | None = None) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def _decision_tree(*, class_weight: str | None = None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def _random_forest(*, class_weight: str | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _xgboost(*, scale_pos_weight: float = 1.0):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to build the XGBoost baseline candidate model. "
            "Install it via `pip install -r requirements-training.txt` or `pip install xgboost`."
        ) from exc
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def train_scale_pos_weight(y_train: pd.Series) -> float:
    """Negatives / positives on the training fold only."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("Training fold has no fraud cases.")
    return negatives / positives


def candidate_pipelines(y_train: pd.Series) -> dict[str, ImbPipeline]:
    """Named candidate pipelines. Names are not rankings."""
    scale_pos_weight = train_scale_pos_weight(y_train)
    return {
        "logistic_regression_smote": build_smote_pipeline(_logistic()),
        "decision_tree_smote": build_smote_pipeline(_decision_tree()),
        "random_forest_smote": build_smote_pipeline(_random_forest()),
        "xgboost_smote": build_smote_pipeline(_xgboost(scale_pos_weight=1.0)),
        "logistic_regression_class_weight": build_class_weight_pipeline(
            _logistic(class_weight="balanced")
        ),
        "decision_tree_class_weight": build_class_weight_pipeline(
            _decision_tree(class_weight="balanced")
        ),
        "random_forest_class_weight": build_class_weight_pipeline(
            _random_forest(class_weight="balanced")
        ),
        "xgboost_class_weight": build_class_weight_pipeline(
            _xgboost(scale_pos_weight=scale_pos_weight)
        ),
    }


def fit_pipeline(pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(pipeline: ImbPipeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path) -> ImbPipeline:
    return joblib.load(path)


def assert_test_set_untouched(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_test_snapshot: pd.DataFrame,
    y_test_snapshot: pd.Series,
) -> None:
    if not X_test.equals(X_test_snapshot):
        raise AssertionError("X_test changed during training.")
    if not y_test.equals(y_test_snapshot):
        raise AssertionError("y_test changed during training.")


def fraud_rate(y: pd.Series) -> float:
    return float((y == 1).mean())


def verify_saved_artifact(path: Path, X_sample: pd.DataFrame) -> dict[str, Any]:
    """Reload a joblib pipeline and run predict / predict_proba on a sample.

    Does not compute evaluation metrics and does not use y_test.
    """
    loaded = load_pipeline(path)
    labels = loaded.predict(X_sample)
    if not hasattr(loaded, "predict_proba"):
        raise AssertionError(f"{path.name} has no predict_proba.")
    probabilities = loaded.predict_proba(X_sample)
    if probabilities.shape != (len(X_sample), 2):
        raise AssertionError(f"{path.name} probability shape is {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise AssertionError(f"{path.name} produced non-finite probabilities.")
    return {
        "path": path.as_posix(),
        "n_sample": int(len(X_sample)),
        "n_predicted": int(len(labels)),
        "probability_shape": list(probabilities.shape),
    }


def _split_stats(split: SplitResult) -> dict[str, Any]:
    return {
        "train_row_count": int(len(split.X_train)),
        "test_row_count": int(len(split.X_test)),
        "train_fraud_count": int((split.y_train == 1).sum()),
        "test_fraud_count": int((split.y_test == 1).sum()),
        "train_fraud_rate": fraud_rate(split.y_train),
        "test_fraud_rate": fraud_rate(split.y_test),
        "overlapping_transaction_ids": int(
            len(set(split.id_train) & set(split.id_test))
        ),
    }


def write_training_metadata(
    *,
    split: SplitResult,
    model_names: list[str],
    artifact_names: list[str],
    scale_pos_weight: float,
    path: Path,
) -> Path:
    payload = {
        "dataset_filename": DATASET_FILENAME,
        "dataset_row_count": int(len(split.X_train) + len(split.X_test)),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_count": len(MODEL_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "numerical_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_features": list(EXCLUDED_FEATURES),
        "smote_strategy": {
            "method": "SMOTE",
            "when": "after train/test split",
            "applied_to": "transformed X_train only",
            "sampling_strategy": "auto",
            "k_neighbors": SMOTE_K_NEIGHBORS,
            "random_state": RANDOM_STATE,
            "one_hot_sparse_output": False,
            "note": "Dense one-hot is used so SMOTE can run on this 5,000-row table.",
        },
        "class_weight_strategy": {
            "sklearn": "class_weight='balanced' for LogisticRegression, DecisionTree, RandomForest",
            "xgboost": "scale_pos_weight = n_neg_train / n_pos_train",
            "scale_pos_weight_train": scale_pos_weight,
        },
        "model_names": model_names,
        "artifact_names": artifact_names,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_split_stats(split),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return path


def write_training_log(
    *,
    split: SplitResult,
    artifact_relpaths: list[str],
    reload_checks: list[dict[str, Any]],
    scale_pos_weight: float,
    path: Path,
) -> Path:
    stats = _split_stats(split)
    lines = [
        "# Phase 5 Training Log",
        "",
        "Candidate baseline pipelines were trained. **No model is declared best.**",
        "Formal comparison belongs to Phase 6. No test-set metrics are reported here.",
        "",
        "## Dataset",
        "",
        f"- File: `data/raw/{DATASET_FILENAME}`",
        f"- Rows after feature engineering: {stats['train_row_count'] + stats['test_row_count']}",
        f"- Target: `{TARGET_COLUMN}`",
        f"- Feature count: {len(MODEL_FEATURES)}",
        "",
        "## Split",
        "",
        f"- test_size: {TEST_SIZE}",
        f"- stratify: y",
        f"- random_state: {RANDOM_STATE}",
        f"- Train rows: {stats['train_row_count']}",
        f"- Test rows: {stats['test_row_count']}",
        f"- Train fraud count / rate: {stats['train_fraud_count']} / {stats['train_fraud_rate'] * 100:.2f}%",
        f"- Test fraud count / rate: {stats['test_fraud_count']} / {stats['test_fraud_rate'] * 100:.2f}%",
        f"- Overlapping Transaction_IDs: {stats['overlapping_transaction_ids']}",
        "",
        "The test fold is not resampled and is not used for fitting or threshold tuning.",
        "",
        "## Feature groups",
        "",
        "Imported from `src.features.feature_engineering` (single source of truth).",
        "",
        f"- Numeric: {', '.join(f'`{c}`' for c in NUMERIC_FEATURES)}",
        f"- Categorical: {', '.join(f'`{c}`' for c in CATEGORICAL_FEATURES)}",
        f"- Excluded: {', '.join(f'`{c}`' for c in EXCLUDED_FEATURES)}",
        "",
        "## Preprocessing",
        "",
        "- `ColumnTransformer`",
        "- Numeric: `RobustScaler` fit on **train only**",
        "- Categorical: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fit on **train only**",
        "- Dense one-hot is intentional: SMOTE cannot use a sparse matrix, and 5,000 rows stay small after encoding.",
        "",
        "## Imbalance strategy",
        "",
        "### Track A — SMOTE",
        "",
        "`imblearn.pipeline.Pipeline`: preprocess -> SMOTE -> model. SMOTE runs only during `fit` on transformed training rows.",
        "",
        "### Track B — class weight",
        "",
        f"No SMOTE. Logistic Regression / Decision Tree / Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight={scale_pos_weight:.4f}` computed from the training fold only.",
        "",
        "## Models trained",
        "",
        "Conservative defaults, `random_state=42`. No grid search.",
        "",
        "- Logistic Regression (`max_iter=2000`)",
        "- Decision Tree (`max_depth=6`, `min_samples_leaf=20`)",
        "- Random Forest (`n_estimators=100`, `max_depth=8`, `min_samples_leaf=10`)",
        "- XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)",
        "",
        "## Artifact locations",
        "",
        *[f"- `{item}`" for item in artifact_relpaths],
        "",
        "## Validation checks",
        "",
        "- X_test and y_test snapshots match after fitting.",
        "- Train/test Transaction_IDs do not overlap.",
        "- Each saved pipeline was reloaded with joblib and `predict_proba` was called on a small X_test sample (labels unused).",
        "",
        "Reload checks:",
        "",
        *[
            f"- `{Path(item['path']).name}`: sample={item['n_sample']}, probability_shape={item['probability_shape']}"
            for item in reload_checks
        ],
        "",
        "Phase 6 will compare models on the untouched test set.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def train_and_save(*, output_dir: Path | None = None) -> dict[str, Any]:
    """Train candidate pipelines and save joblib artifacts. No winner is chosen."""
    set_global_seed(RANDOM_STATE)
    output_dir = output_dir or BASELINE_MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    x_test_snapshot = split.X_test.copy(deep=True)
    y_test_snapshot = split.y_test.copy(deep=True)

    pipelines = candidate_pipelines(split.y_train)
    sample = split.X_test.head(8)
    reload_checks: list[dict[str, Any]] = []
    artifact_names: list[str] = []
    artifact_relpaths: list[str] = []

    for name, pipeline in pipelines.items():
        fit_pipeline(pipeline, split.X_train, split.y_train)
        assert_test_set_untouched(
            split.X_test, split.y_test, x_test_snapshot, y_test_snapshot
        )
        if not hasattr(pipeline, "predict_proba"):
            raise AssertionError(f"{name} does not expose predict_proba.")
        pipeline.predict_proba(sample)
        filename = f"{name}.joblib"
        path = output_dir / filename
        save_pipeline(pipeline, path)
        check = verify_saved_artifact(path, sample)
        reload_checks.append(check)
        artifact_names.append(filename)
        try:
            artifact_relpaths.append(path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix())
        except ValueError:
            artifact_relpaths.append(path.as_posix())

    assert_test_set_untouched(split.X_test, split.y_test, x_test_snapshot, y_test_snapshot)
    scale_pos_weight = train_scale_pos_weight(split.y_train)

    metadata_path = output_dir / "training_metadata.json"
    write_training_metadata(
        split=split,
        model_names=list(pipelines.keys()),
        artifact_names=artifact_names,
        scale_pos_weight=scale_pos_weight,
        path=metadata_path,
    )
    log_path = REPORTS_DIR / "training_log.md"
    write_training_log(
        split=split,
        artifact_relpaths=artifact_relpaths + ["models/baseline/training_metadata.json"],
        reload_checks=reload_checks,
        scale_pos_weight=scale_pos_weight,
        path=log_path,
    )
    return {
        "split": _split_stats(split),
        "artifacts": artifact_relpaths,
        "metadata_path": str(metadata_path),
        "log_path": str(log_path),
        "scale_pos_weight": scale_pos_weight,
        "model_names": list(pipelines.keys()),
    }


def build_pipeline(*, strategy: str = "smote", estimator: Any | None = None) -> ImbPipeline:
    """Public constructor used by tests and later serving code."""
    model = estimator or _logistic()
    if strategy == "smote":
        return build_smote_pipeline(model)
    if strategy == "class_weight":
        if hasattr(model, "set_params") and "class_weight" in model.get_params():
            model.set_params(class_weight="balanced")
        return build_class_weight_pipeline(model)
    raise ValueError("strategy must be 'smote' or 'class_weight'")
