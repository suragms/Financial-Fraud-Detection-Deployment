"""Phase 9 integration, inference-edge, leakage, and reproducibility tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.data_loader import (
    EXPECTED_RAW_DATASET_MD5,
    FORBIDDEN_MODEL_INPUTS,
    dataset_hash,
    export_scored_csv,
    load_metadata,
    load_pipeline,
    load_raw_dataset,
    score_dataset,
)
from src.data.loader import ValidationError
from src.features.feature_engineering import EXCLUDED_FEATURES, FEATURE_INPUT_COLUMNS, MODEL_FEATURES
from src.models.predict import predict_records, predict_transaction, prepare_features
from src.models.scoring import VALID_RISK_BAND_NAMES, assign_risk_band, probability_to_risk_score
from src.utils.paths import (
    BASELINE_MODELS_DIR,
    DASHBOARD_DIR,
    PRODUCTION_MODELS_DIR,
    PROJECT_ROOT,
    RAW_DATASET_PATH,
    REPORTS_DIR,
)

EXPECTED_PRODUCTION_MODEL_MD5 = "5d08c124ddbb452f7a93f6982d02240a"
REQUIRED_PATHS = [
    RAW_DATASET_PATH,
    PROJECT_ROOT / "src" / "data",
    PROJECT_ROOT / "src" / "features",
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "src" / "evaluation",
    PROJECT_ROOT / "src" / "utils",
    BASELINE_MODELS_DIR,
    PRODUCTION_MODELS_DIR,
    DASHBOARD_DIR,
    REPORTS_DIR,
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "tests",
    PRODUCTION_MODELS_DIR / "final_fraud_pipeline.joblib",
    PRODUCTION_MODELS_DIR / "model_metadata.json",
    DASHBOARD_DIR / "app.py",
    DASHBOARD_DIR / "components.py",
    DASHBOARD_DIR / "data_loader.py",
    DASHBOARD_DIR / "styles.py",
]


def _valid_record() -> dict:
    raw = load_raw_dataset().iloc[0]
    return {column: raw[column] for column in FEATURE_INPUT_COLUMNS}


def test_required_project_files_exist() -> None:
    missing = [path.as_posix() for path in REQUIRED_PATHS if not path.exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_raw_and_production_hashes_frozen() -> None:
    import hashlib

    assert dataset_hash() == EXPECTED_RAW_DATASET_MD5
    model_digest = hashlib.md5(
        (PRODUCTION_MODELS_DIR / "final_fraud_pipeline.joblib").read_bytes()
    ).hexdigest()
    assert model_digest == EXPECTED_PRODUCTION_MODEL_MD5


def test_production_pipeline_integrity() -> None:
    metadata = load_metadata()
    pipeline = load_pipeline()
    assert float(metadata["production_threshold"]) == pytest.approx(0.10)
    assert float(pipeline.threshold) == pytest.approx(0.10)
    assert metadata["model_name"] == "Logistic Regression"
    assert metadata["strategy"] == "Class Weight"
    assert str(metadata["calibration_method"]).lower() == "sigmoid"
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "predict")
    sample = load_raw_dataset().head(3).loc[:, list(FEATURE_INPUT_COLUMNS)]
    proba = pipeline.predict_proba(prepare_features(sample))
    assert proba.shape == (3, 2)
    first = pipeline.score(prepare_features(sample))
    second = pipeline.score(prepare_features(sample))
    pd.testing.assert_frame_equal(first, second)
    for forbidden in EXCLUDED_FEATURES:
        assert forbidden not in metadata["feature_list"]
        assert forbidden not in MODEL_FEATURES


def test_predict_transaction_and_records() -> None:
    pipeline = load_pipeline()
    record = _valid_record()
    one = predict_transaction(record, pipeline=pipeline)
    many = predict_records(pd.DataFrame([record, record]), pipeline=pipeline)
    assert len(many) == 2
    assert one["predicted_label"] in (0, 1)
    assert math.isfinite(one["predicted_probability"])


@pytest.mark.parametrize(
    "field",
    ["Transaction_Amount", "Merchant_Category", "Transaction_Date"],
)
def test_missing_required_field_fails_safely(field: str) -> None:
    record = _valid_record()
    record.pop(field)
    with pytest.raises(ValidationError, match="missing required fields"):
        predict_transaction(record, pipeline=load_pipeline())


def test_invalid_numeric_and_date_fail_safely() -> None:
    pipeline = load_pipeline()
    bad_numeric = _valid_record()
    bad_numeric["Transaction_Amount"] = "not-a-number"
    with pytest.raises(ValidationError):
        predict_transaction(bad_numeric, pipeline=pipeline)
    negative = _valid_record()
    negative["Transaction_Amount"] = -12.5
    with pytest.raises(ValidationError, match="cannot be negative"):
        predict_transaction(negative, pipeline=pipeline)
    nan_row = _valid_record()
    nan_row["Average_Spend"] = float("nan")
    with pytest.raises(ValidationError):
        predict_transaction(nan_row, pipeline=pipeline)
    inf_row = _valid_record()
    inf_row["Account_Age_Days"] = float("inf")
    with pytest.raises(ValidationError):
        predict_transaction(inf_row, pipeline=pipeline)
    bad_date = _valid_record()
    bad_date["Transaction_Date"] = "2023/13/99 99:99"
    with pytest.raises(ValidationError):
        predict_transaction(bad_date, pipeline=pipeline)


def test_zero_amount_is_accepted() -> None:
    record = _valid_record()
    record["Transaction_Amount"] = 0.0
    result = predict_transaction(record, pipeline=load_pipeline())
    assert math.isfinite(result["predicted_probability"])


def test_unknown_category_does_not_crash() -> None:
    record = _valid_record()
    record["Merchant_Category"] = "Unknown_QA_Category"
    result = predict_transaction(record, pipeline=load_pipeline())
    assert result["risk_band"] in VALID_RISK_BAND_NAMES


def test_empty_dataframe_fails_safely() -> None:
    empty = pd.DataFrame(columns=list(FEATURE_INPUT_COLUMNS))
    with pytest.raises(ValidationError, match="empty"):
        predict_records(empty, pipeline=load_pipeline())


def test_risk_score_and_threshold_boundaries() -> None:
    pipeline = load_pipeline()
    threshold = float(pipeline.threshold)
    assert threshold == pytest.approx(0.10)
    probabilities = np.array([0.00, 0.09, 0.099999, 0.10, 0.100001, 0.50, 1.00])
    scores = probability_to_risk_score(probabilities)
    assert np.all(scores >= 0.0) and np.all(scores <= 100.0)
    np.testing.assert_allclose(scores, probabilities * 100.0)
    flagged = probabilities >= threshold
    assert flagged.tolist() == [False, False, False, True, True, True, True]
    bands = assign_risk_band(scores, pipeline.risk_bands)
    assert bands[0] == "Low Risk"
    assert bands[3] == "Medium Risk"
    assert set(bands).issubset(set(VALID_RISK_BAND_NAMES))


def test_inference_is_deterministic() -> None:
    pipeline = load_pipeline()
    record = _valid_record()
    first = predict_transaction(record, pipeline=pipeline)
    second = predict_transaction(record, pipeline=pipeline)
    assert first == second


def test_score_dataset_does_not_pass_forbidden_model_inputs() -> None:
    raw = load_raw_dataset()
    snapshot = raw.copy(deep=True)
    scored = score_dataset(raw, load_pipeline())
    pd.testing.assert_frame_equal(raw, snapshot)
    for forbidden in ("Customer_ID", "Suspicious_Keyword"):
        assert forbidden not in scored.columns
    assert "Historical Ground Truth" in scored.columns
    assert "Fraudulent" not in scored.columns
    features = prepare_features(raw.head(2))
    for forbidden in FORBIDDEN_MODEL_INPUTS:
        assert forbidden not in features.columns
    assert "Transaction_Date" not in features.columns


def test_csv_export_10_and_full(tmp_path: Path) -> None:
    pipeline = load_pipeline()
    raw = load_raw_dataset()
    ten = score_dataset(raw.head(10), pipeline)
    full = score_dataset(raw, pipeline)
    ten_path = export_scored_csv(ten, tmp_path / "ten.csv")
    full_path = export_scored_csv(full, tmp_path / "full.csv")
    ten_loaded = pd.read_csv(ten_path)
    full_loaded = pd.read_csv(full_path)
    assert len(ten_loaded) == 10
    assert len(full_loaded) == 5000
    for column in ["predicted_probability", "risk_score", "risk_band", "predicted_label"]:
        assert column in ten_loaded.columns
        assert column in full_loaded.columns
    assert "Customer_ID" not in full_loaded.columns
    assert "Suspicious_Keyword" not in full_loaded.columns
    assert dataset_hash() == EXPECTED_RAW_DATASET_MD5
    with pytest.raises(ValueError, match="raw dataset"):
        export_scored_csv(ten, RAW_DATASET_PATH)


def test_dashboard_source_does_not_retrain() -> None:
    text = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8") + (
        DASHBOARD_DIR / "data_loader.py"
    ).read_text(encoding="utf-8")
    assert ".fit(" not in text
    assert "train_and_save" not in text
    assert "CalibratedClassifierCV" not in text
    assert "SMOTE(" not in text


def test_streamlit_prediction_and_monitoring_workflow() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=90)
    app.run()
    assert not app.exception
    for page in (
        "Overview",
        "Fraud Analytics",
        "Model Performance",
        "Transaction Prediction",
        "Risk Monitoring",
    ):
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception
    app.sidebar.radio[0].set_value("Transaction Prediction")
    app.run()
    submit = [button for button in app.button if "Score transaction" in button.label]
    assert submit
    submit[0].click()
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Risk Monitoring")
    app.run()
    assert not app.exception
    assert any("Download scored results CSV" in button.label for button in app.download_button)
