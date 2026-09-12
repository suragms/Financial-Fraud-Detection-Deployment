"""Phase 7 production pipeline, metadata, inference, and split-integrity tests."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.data.eda import raw_dataset_md5
from src.data.loader import load_fraud_dataset
from src.evaluation.compare import (
    EXPECTED_RAW_DATASET_MD5,
    EXPECTED_TEST_FRAUD_COUNT,
    EXPECTED_TEST_ROWS,
    assert_test_fold,
)
from src.features.feature_engineering import EXCLUDED_FEATURES, MODEL_FEATURES
from src.models.pipeline import load_xy, make_stratified_split
from src.models.predict import (
    PRODUCTION_METADATA_PATH,
    PRODUCTION_PIPELINE_PATH,
    load_production_pipeline,
    predict_records,
    predict_transaction,
    prepare_features,
)
from src.models.scoring import VALID_RISK_BAND_NAMES
from src.utils.seeds import RANDOM_STATE


@pytest.fixture(scope="module")
def production():
    if not PRODUCTION_PIPELINE_PATH.exists() or not PRODUCTION_METADATA_PATH.exists():
        from src.models.phase7 import run_phase7

        run_phase7()
    return load_production_pipeline()


@pytest.fixture(scope="module")
def metadata(production) -> dict:
    return json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split():
    X, y, ids = load_xy()
    return make_stratified_split(X, y, ids)


def test_production_artifact_exists(production) -> None:
    assert PRODUCTION_PIPELINE_PATH.exists()
    assert PRODUCTION_PIPELINE_PATH.is_file()
    assert production is not None


def test_production_artifact_loads(production) -> None:
    assert production is not None
    assert hasattr(production, "predict_proba")
    assert hasattr(production, "predict")
    assert hasattr(production, "score")


def test_metadata_exists_and_is_valid(metadata) -> None:
    assert PRODUCTION_METADATA_PATH.exists()
    required = [
        "model_name",
        "strategy",
        "feature_list",
        "preprocessing",
        "calibration_method",
        "production_threshold",
        "risk_score_method",
        "risk_bands",
        "random_state",
        "training_row_count",
        "training_fraud_count",
        "training_legitimate_count",
        "test_row_count",
        "phase6_reference_metrics",
        "model_version",
        "created_at_utc",
    ]
    missing = [key for key in required if key not in metadata]
    assert not missing, f"Missing metadata keys: {missing}"
    assert metadata["random_state"] == RANDOM_STATE
    assert metadata["training_row_count"] == 4000
    assert metadata["training_fraud_count"] == 386
    assert metadata["training_legitimate_count"] == 3614
    assert metadata["test_row_count"] == 1000


def test_required_model_features_are_correct(metadata) -> None:
    assert list(metadata["feature_list"]) == list(MODEL_FEATURES)
    assert metadata["feature_count"] == len(MODEL_FEATURES)


@pytest.mark.parametrize(
    "forbidden",
    ["Customer_ID", "Transaction_ID", "Suspicious_Keyword", "Fraudulent", "Transaction_Date"],
)
def test_forbidden_columns_are_not_model_features(metadata, forbidden: str) -> None:
    assert forbidden not in metadata["feature_list"]
    assert forbidden in EXCLUDED_FEATURES


def test_risk_score_between_0_and_100(production, split) -> None:
    scored = production.score(split.X_test.head(25))
    scores = scored["risk_score"].to_numpy()
    assert np.isfinite(scores).all()
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 100.0


def test_risk_bands_are_valid(production, split, metadata) -> None:
    scored = production.score(split.X_test.head(50))
    assert set(scored["risk_band"]).issubset(set(VALID_RISK_BAND_NAMES))
    band_names = [band["name"] for band in metadata["risk_bands"]]
    assert band_names == list(VALID_RISK_BAND_NAMES)


def test_threshold_is_between_0_and_1(production, metadata) -> None:
    assert 0.0 < float(production.threshold) < 1.0
    assert 0.0 < float(metadata["production_threshold"]) < 1.0
    assert production.threshold == pytest.approx(metadata["production_threshold"])


def test_inference_returns_probability_score_band_label() -> None:
    raw = load_fraud_dataset().head(1)
    result = predict_transaction(raw)
    assert set(result) == {
        "predicted_probability",
        "risk_score",
        "risk_band",
        "predicted_label",
    }
    assert result["risk_band"] in VALID_RISK_BAND_NAMES
    assert result["predicted_label"] in (0, 1)
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 100.0


def test_probability_is_finite(production, split) -> None:
    proba = production.predict_proba(split.X_test.head(20))
    assert proba.shape == (20, 2)
    assert np.isfinite(proba).all()


def test_prediction_is_deterministic(production, split) -> None:
    sample = split.X_test.head(15)
    first = production.score(sample)
    second = production.score(sample)
    pd.testing.assert_frame_equal(first, second)
    raw = load_fraud_dataset().head(1)
    a = predict_transaction(raw, pipeline=production)
    b = predict_transaction(raw, pipeline=production)
    assert a == b


def test_test_set_remains_1000_96_904(split) -> None:
    info = assert_test_fold(split)
    assert info["test_rows"] == EXPECTED_TEST_ROWS
    assert info["test_fraud_count"] == EXPECTED_TEST_FRAUD_COUNT
    assert info["test_legitimate_count"] == 904
    assert int((split.y_test == 1).sum()) == 96
    assert int((split.y_test == 0).sum()) == 904
    assert len(split.X_test) == 1000


def test_raw_dataset_hash_unchanged() -> None:
    assert raw_dataset_md5() == EXPECTED_RAW_DATASET_MD5


def test_prepare_features_rejects_missing_fields() -> None:
    with pytest.raises(Exception):
        prepare_features({"Transaction_Amount": 10})


def test_inference_ignores_non_feature_identifiers() -> None:
    raw = load_fraud_dataset().head(2).copy()
    features = prepare_features(raw)
    assert "Customer_ID" not in features.columns
    assert "Transaction_ID" not in features.columns
    assert "Suspicious_Keyword" not in features.columns
    assert "Fraudulent" not in features.columns
    assert "Transaction_Date" not in features.columns
    scored = predict_records(raw)
    assert len(scored) == 2
    assert math.isfinite(float(scored.iloc[0]["predicted_probability"]))
