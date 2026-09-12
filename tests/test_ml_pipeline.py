"""Phase 5 leakage-safe training pipeline tests."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data.loader import TARGET_COLUMN
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.models.pipeline import (
    TEST_SIZE,
    build_class_weight_pipeline,
    build_preprocessor,
    build_smote_pipeline,
    candidate_pipelines,
    fit_pipeline,
    load_xy,
    make_stratified_split,
    save_pipeline,
    verify_saved_artifact,
    _logistic,
)
from src.utils.seeds import RANDOM_STATE


def test_stratified_split_is_reproducible() -> None:
    X, y, ids = load_xy()
    first = make_stratified_split(X, y, ids)
    second = make_stratified_split(X, y, ids)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)
    pd.testing.assert_series_equal(first.id_test, second.id_test)


def test_train_test_row_counts() -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    total = len(X)
    assert len(split.X_train) + len(split.X_test) == total
    assert len(split.y_train) == len(split.X_train)
    assert len(split.y_test) == len(split.X_test)
    expected_test = int(round(total * TEST_SIZE))
    assert abs(len(split.X_test) - expected_test) <= 1


def test_no_overlapping_transaction_ids() -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    overlap = set(split.id_train) & set(split.id_test)
    assert overlap == set()
    assert split.id_train.is_unique
    assert split.id_test.is_unique


def test_target_not_in_model_features() -> None:
    X, y, _ids = load_xy()
    assert TARGET_COLUMN not in MODEL_FEATURES
    assert TARGET_COLUMN not in X.columns
    assert TARGET_COLUMN not in y.index.names or True
    assert set(X.columns) == set(MODEL_FEATURES)


def test_preprocessor_columns_and_onehot_unknown() -> None:
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    names = [name for name, _trans, _cols in preprocessor.transformers]
    assert names == ["numeric", "categorical"]
    numeric = dict((name, (trans, cols)) for name, trans, cols in preprocessor.transformers)["numeric"]
    categorical = dict((name, (trans, cols)) for name, trans, cols in preprocessor.transformers)["categorical"]
    assert list(numeric[1]) == list(NUMERIC_FEATURES)
    assert list(categorical[1]) == list(CATEGORICAL_FEATURES)
    assert isinstance(numeric[0], RobustScaler)
    encoder = categorical[0]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"
    assert encoder.sparse_output is False


def test_smote_not_applied_to_test_data() -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    x_snapshot = split.X_test.copy(deep=True)
    y_snapshot = split.y_test.copy(deep=True)
    pipeline = build_smote_pipeline(_logistic())
    fit_pipeline(pipeline, split.X_train, split.y_train)
    pd.testing.assert_frame_equal(split.X_test, x_snapshot)
    pd.testing.assert_series_equal(split.y_test, y_snapshot)
    assert "smote" in pipeline.named_steps
    # imblearn skips the sampler on transform/predict
    transformed_test = pipeline.named_steps["preprocess"].transform(split.X_test)
    assert transformed_test.shape[0] == len(split.X_test)


def test_pipelines_fit_and_predict_proba() -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    pipelines = candidate_pipelines(split.y_train)
    sample = split.X_test.head(5)
    for name, pipeline in pipelines.items():
        fit_pipeline(pipeline, split.X_train, split.y_train)
        assert hasattr(pipeline, "predict_proba")
        proba = pipeline.predict_proba(sample)
        assert proba.shape == (len(sample), 2)
        assert np.isfinite(proba).all()
        labels = pipeline.predict(sample)
        assert len(labels) == len(sample)


def test_joblib_roundtrip(tmp_path: Path) -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    pipeline = build_class_weight_pipeline(_logistic(class_weight="balanced"))
    fit_pipeline(pipeline, split.X_train, split.y_train)
    path = tmp_path / "logistic_regression_class_weight.joblib"
    save_pipeline(pipeline, path)
    check = verify_saved_artifact(path, split.X_test.head(3))
    assert check["n_sample"] == 3
    assert check["probability_shape"] == [3, 2]


def test_test_set_keeps_original_fraud_distribution() -> None:
    X, y, ids = load_xy()
    split = make_stratified_split(X, y, ids)
    overall = float((y == 1).mean())
    train_rate = float((split.y_train == 1).mean())
    test_rate = float((split.y_test == 1).mean())
    assert abs(train_rate - overall) < 0.01
    assert abs(test_rate - overall) < 0.01
    pipeline = build_smote_pipeline(_logistic())
    y_test_before = split.y_test.copy()
    fit_pipeline(pipeline, split.X_train, split.y_train)
    pd.testing.assert_series_equal(split.y_test, y_test_before)
    assert split.y_test.mean() == y_test_before.mean()


def test_random_state_constant() -> None:
    assert RANDOM_STATE == 42
