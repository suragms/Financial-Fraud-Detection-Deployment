"""Phase 6 evaluation tests. Scores are not hard-coded."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.eda import raw_dataset_md5
from src.evaluation.compare import (
    CANDIDATES,
    DISPLAY_COLS,
    EXPECTED_RAW_DATASET_MD5,
    EXPECTED_TEST_FRAUD_COUNT,
    EXPECTED_TEST_FRAUD_RATE_PCT,
    EXPECTED_TEST_ROWS,
    assert_raw_dataset_unchanged,
    assert_test_fold,
    candidate_artifact_path,
    evaluate_candidates,
    load_candidates,
    load_evaluation_split,
    select_candidate,
    write_comparison_csv,
    write_evaluation_charts,
    write_selected_model_json,
)
from src.evaluation.metrics import compute_binary_metrics, dummy_always_legitimate
from src.models.pipeline import load_xy, make_stratified_split


METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]


@pytest.fixture(scope="module")
def split():
    return load_evaluation_split()


@pytest.fixture(scope="module")
def pipelines():
    return load_candidates()


@pytest.fixture(scope="module")
def scored(split, pipelines):
    comparison, curves = evaluate_candidates(split, pipelines)
    return comparison, curves


@pytest.fixture(scope="module")
def trained(scored):
    comparison, _curves = scored
    return comparison[comparison["Strategy"] != "Dummy"].copy()


def test_raw_dataset_hash_unchanged() -> None:
    digest = raw_dataset_md5()
    assert digest == EXPECTED_RAW_DATASET_MD5
    assert assert_raw_dataset_unchanged() == digest


def test_all_eight_artifacts_load(pipelines) -> None:
    assert len(CANDIDATES) == 8
    assert set(pipelines) == {key for key, _name, _strategy in CANDIDATES}
    for key, _name, _strategy in CANDIDATES:
        path = candidate_artifact_path(key)
        assert path.exists()
        assert hasattr(pipelines[key], "predict")
        assert hasattr(pipelines[key], "predict_proba")


def test_test_fold_matches_phase_5(split) -> None:
    info = assert_test_fold(split)
    assert info["test_rows"] == EXPECTED_TEST_ROWS
    assert info["test_fraud_count"] == EXPECTED_TEST_FRAUD_COUNT
    assert round(info["test_fraud_rate"] * 100, 2) == EXPECTED_TEST_FRAUD_RATE_PCT
    assert len(split.X_test) == EXPECTED_TEST_ROWS
    assert int((split.y_test == 1).sum()) == EXPECTED_TEST_FRAUD_COUNT
    assert round(float((split.y_test == 1).mean()) * 100, 2) == 9.60


def test_split_is_recreated_identically() -> None:
    first = load_evaluation_split()
    X, y, ids = load_xy()
    second = make_stratified_split(X, y, ids)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_all_eight_produce_predictions_and_probabilities(split, scored, trained) -> None:
    comparison, curves = scored
    n = len(split.X_test)
    for key, _name, _strategy in CANDIDATES:
        y_pred = curves[key]["y_pred"]
        y_proba = curves[key]["y_proba"]
        assert len(y_pred) == n
        assert len(y_proba) == n
        assert y_pred.dtype.kind in {"i", "u"}
        assert np.isin(y_pred, [0, 1]).all()
        assert np.isfinite(y_proba).all()
        assert float(y_proba.min()) >= 0.0
        assert float(y_proba.max()) <= 1.0
    assert len(trained) == 8


def test_confusion_matrix_totals_equal_test_size(trained) -> None:
    totals = trained["TN"] + trained["FP"] + trained["FN"] + trained["TP"]
    assert (totals == EXPECTED_TEST_ROWS).all()


def test_metrics_are_finite_without_nan_or_inf(trained) -> None:
    for column in METRIC_COLUMNS:
        values = trained[column].to_numpy(dtype="float64")
        assert np.isfinite(values).all()
        assert not np.isnan(values).any()


def test_auc_bounds(trained) -> None:
    assert (trained["ROC_AUC"] >= 0.0).all()
    assert (trained["ROC_AUC"] <= 1.0).all()
    assert (trained["PR_AUC"] >= 0.0).all()
    assert (trained["PR_AUC"] <= 1.0).all()


def test_roc_auc_uses_probabilities_not_hard_labels(split, scored) -> None:
    _comparison, curves = scored
    y_test = split.y_test
    for key, _name, _strategy in CANDIDATES:
        y_proba = curves[key]["y_proba"]
        y_pred = curves[key]["y_pred"]
        auc_from_proba = roc_auc_score(y_test, y_proba)
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        assert metrics["ROC_AUC"] == pytest.approx(auc_from_proba)
        assert metrics["PR_AUC"] == pytest.approx(average_precision_score(y_test, y_proba))


def test_same_test_distribution_for_every_candidate(split, pipelines, scored) -> None:
    comparison, curves = scored
    trained = comparison[comparison["Strategy"] != "Dummy"]
    supports = trained["support"].unique()
    assert list(supports) == [EXPECTED_TEST_ROWS]
    first_pred_len = len(curves[CANDIDATES[0][0]]["y_pred"])
    for key, _name, _strategy in CANDIDATES:
        assert len(curves[key]["y_pred"]) == first_pred_len
        assert len(curves[key]["y_proba"]) == first_pred_len
    x_snapshot = split.X_test.copy(deep=True)
    y_snapshot = split.y_test.copy(deep=True)
    for pipeline in pipelines.values():
        pipeline.predict(split.X_test.head(3))
    pd.testing.assert_frame_equal(split.X_test, x_snapshot)
    pd.testing.assert_series_equal(split.y_test, y_snapshot)


def test_evaluation_does_not_mutate_test_labels(split, pipelines) -> None:
    x_before = split.X_test.copy(deep=True)
    y_before = split.y_test.copy(deep=True)
    evaluate_candidates(split, pipelines)
    pd.testing.assert_frame_equal(split.X_test, x_before)
    pd.testing.assert_series_equal(split.y_test, y_before)


def test_metrics_are_reproducible(split, pipelines) -> None:
    first, _ = evaluate_candidates(split, pipelines)
    second, _ = evaluate_candidates(split, pipelines)
    left = first[first["Strategy"] != "Dummy"][DISPLAY_COLS + ["key"]].reset_index(drop=True)
    right = second[second["Strategy"] != "Dummy"][DISPLAY_COLS + ["key"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_dummy_always_legitimate_shows_accuracy_trap(split) -> None:
    dummy = dummy_always_legitimate(split.y_test)
    assert dummy["Recall"] == 0.0
    assert dummy["TP"] == 0
    assert dummy["FN"] == EXPECTED_TEST_FRAUD_COUNT
    assert dummy["FP"] == 0
    assert dummy["TN"] == EXPECTED_TEST_ROWS - EXPECTED_TEST_FRAUD_COUNT
    assert dummy["Accuracy"] == pytest.approx(
        (EXPECTED_TEST_ROWS - EXPECTED_TEST_FRAUD_COUNT) / EXPECTED_TEST_ROWS
    )


def test_selected_candidate_is_a_trained_model(scored) -> None:
    comparison, _curves = scored
    selected = select_candidate(comparison)
    assert selected["Strategy"] != "Dummy"
    assert selected["key"] in {key for key, _name, _strategy in CANDIDATES}
    for column in METRIC_COLUMNS:
        assert math.isfinite(float(selected[column]))


def test_writers_use_computed_values(tmp_path, split, scored) -> None:
    comparison, curves = scored
    csv_path = write_comparison_csv(comparison, tmp_path / "model_comparison.csv")
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == DISPLAY_COLS
    assert len(loaded) == 8
    assert "Always Legitimate" not in set(loaded["Model"])
    trained = comparison[comparison["Strategy"] != "Dummy"].reset_index(drop=True)
    assert loaded["Recall"].tolist() == pytest.approx(trained["Recall"].tolist())
    selected = select_candidate(comparison)
    json_path = write_selected_model_json(selected, tmp_path / "selected_model.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "selected candidate"
    assert payload["production_ready"] is False
    assert payload["test_metrics"]["F1"] == pytest.approx(float(selected["F1"]))
    charts = write_evaluation_charts(comparison, curves, split.y_test, tmp_path / "charts")
    for path in charts.values():
        assert path.exists()
        assert path.stat().st_size > 0
