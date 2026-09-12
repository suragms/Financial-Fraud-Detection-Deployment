"""Dashboard application-layer tests. They do not modify the raw dataset."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from dashboard.data_loader import (
    EXPECTED_DATASET_ROWS,
    EXPECTED_RAW_DATASET_MD5,
    FORBIDDEN_MODEL_INPUTS,
    dataset_hash,
    load_final_test_metrics,
    load_metadata,
    load_pipeline,
    load_raw_dataset,
    score_dataset,
)
from src.data.loader import DATE_FORMAT
from src.features.feature_engineering import FEATURE_INPUT_COLUMNS
from src.models.predict import predict_transaction
from src.models.scoring import VALID_RISK_BAND_NAMES
from src.utils.paths import RAW_DATASET_PATH


def test_dashboard_modules_import() -> None:
    import dashboard.app as app
    import dashboard.components as components
    import dashboard.data_loader as data_loader
    import dashboard.styles as styles

    assert app.PAGES[-1] == "Risk Monitoring"
    assert components.inject_styles is not None
    assert data_loader.load_raw_dataset is not None
    assert "kpi-card" in styles.CSS


def test_production_model_loads() -> None:
    pipeline = load_pipeline()
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "score")


def test_metadata_loads() -> None:
    metadata = load_metadata()
    threshold = float(metadata["production_threshold"])
    assert 0.0 < threshold < 1.0
    assert metadata["model_name"] == "Logistic Regression"


def test_dataset_loads_and_row_count_is_5000() -> None:
    frame = load_raw_dataset()
    assert len(frame) == EXPECTED_DATASET_ROWS
    assert len(frame) == 5000


def test_raw_dataset_hash_unchanged() -> None:
    assert dataset_hash() == EXPECTED_RAW_DATASET_MD5


def test_prediction_function_works() -> None:
    raw = load_raw_dataset().iloc[[0]]
    record = {column: raw.iloc[0][column] for column in FEATURE_INPUT_COLUMNS}
    result = predict_transaction(record, pipeline=load_pipeline())
    assert math.isfinite(result["predicted_probability"])
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_band"] in VALID_RISK_BAND_NAMES
    assert result["predicted_label"] in (0, 1)


def test_required_prediction_fields_are_accepted() -> None:
    raw = load_raw_dataset().iloc[0]
    for forbidden in FORBIDDEN_MODEL_INPUTS:
        assert forbidden not in FEATURE_INPUT_COLUMNS
    record = {
        "Transaction_Date": "15-06-2023 14:30",
        "Transaction_Amount": 100.0,
        "Average_Spend": 80.0,
        "Previous_Transactions": 5,
        "Account_Age_Days": 200,
        "Is_International": 0,
        "Merchant_Category": raw["Merchant_Category"],
        "Payment_Method": raw["Payment_Method"],
        "Device_Type": raw["Device_Type"],
        "Location": raw["Location"],
    }
    datetime.strptime(record["Transaction_Date"], DATE_FORMAT)
    result = predict_transaction(record, pipeline=load_pipeline())
    assert math.isfinite(result["predicted_probability"])


def test_probability_finite_and_risk_bounds() -> None:
    scored = score_dataset(load_raw_dataset().head(20), load_pipeline())
    assert np.isfinite(scored["predicted_probability"].to_numpy()).all()
    assert float(scored["risk_score"].min()) >= 0.0
    assert float(scored["risk_score"].max()) <= 100.0
    assert set(scored["risk_band"]).issubset(set(VALID_RISK_BAND_NAMES))


def test_threshold_between_0_and_1() -> None:
    metadata = load_metadata()
    pipeline = load_pipeline()
    assert 0.0 < float(metadata["production_threshold"]) < 1.0
    assert 0.0 < float(pipeline.threshold) < 1.0


def test_csv_export_does_not_write_raw_dataset(tmp_path) -> None:
    scored = score_dataset(load_raw_dataset().head(5), load_pipeline())
    export_path = tmp_path / "scored_transactions.csv"
    scored.to_csv(export_path, index=False)
    assert export_path.exists()
    assert dataset_hash() == EXPECTED_RAW_DATASET_MD5
    assert RAW_DATASET_PATH.exists()


def test_final_metrics_file_is_readable() -> None:
    metrics = load_final_test_metrics()
    assert "Recall" in metrics
    assert "Precision" in metrics


def test_streamlit_app_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    from src.utils.paths import PROJECT_ROOT

    script = PROJECT_ROOT / "dashboard" / "app.py"
    app = AppTest.from_file(str(script), default_timeout=90)
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Fraud Analytics")
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Model Performance")
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Transaction Prediction")
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Risk Monitoring")
    app.run()
    assert not app.exception
    assert any(button.label == "Download scored results CSV" for button in app.download_button)
