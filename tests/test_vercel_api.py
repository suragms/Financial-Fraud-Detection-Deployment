"""Vercel Flask API tests. They do not retrain or modify frozen artifacts."""

from __future__ import annotations

import math

import pytest

from src.models.scoring import VALID_RISK_BAND_NAMES

LOW_RISK_PAYLOAD = {
    "Transaction_Date": "15-06-2023 14:30",
    "Transaction_Amount": 50.0,
    "Average_Spend": 95.0,
    "Previous_Transactions": 20,
    "Account_Age_Days": 800,
    "Is_International": 0,
    "Merchant_Category": "Food",
    "Payment_Method": "Debit Card",
    "Device_Type": "Mobile",
    "Location": "Mumbai",
}

HIGH_RISK_PAYLOAD = {
    "Transaction_Date": "02-01-2023 02:15",
    "Transaction_Amount": 250.0,
    "Average_Spend": 40.0,
    "Previous_Transactions": 1,
    "Account_Age_Days": 15,
    "Is_International": 1,
    "Merchant_Category": "Travel",
    "Payment_Method": "Credit Card",
    "Device_Type": "POS",
    "Location": "Delhi",
}


@pytest.fixture(scope="module")
def client():
    from api.index import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _assert_prediction_schema(payload: dict) -> None:
    assert "predicted_probability" in payload
    assert "risk_score" in payload
    assert "risk_band" in payload
    assert "predicted_label" in payload
    assert "flagged_for_review" in payload
    assert "threshold" in payload
    assert math.isfinite(float(payload["predicted_probability"]))
    assert 0.0 <= float(payload["risk_score"]) <= 100.0
    assert payload["risk_band"] in VALID_RISK_BAND_NAMES
    assert payload["predicted_label"] in (0, 1)
    assert payload["flagged_for_review"] in (True, False)
    assert float(payload["threshold"]) == pytest.approx(0.10)


def test_health_endpoint(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "Logistic Regression + Class Weight"
    assert float(body["threshold"]) == pytest.approx(0.10)
    assert "error" not in body
    assert "traceback" not in str(body).lower()


def test_valid_low_risk_prediction(client) -> None:
    response = client.post("/api/predict", json=LOW_RISK_PAYLOAD)
    assert response.status_code == 200
    body = response.get_json()
    _assert_prediction_schema(body)
    threshold = float(body["threshold"])
    if body["predicted_probability"] < threshold:
        assert body["predicted_label"] == 0
        assert body["flagged_for_review"] is False
        assert body["risk_band"] == "Low Risk"


def test_valid_high_risk_prediction(client) -> None:
    response = client.post("/api/predict", json=HIGH_RISK_PAYLOAD)
    assert response.status_code == 200
    body = response.get_json()
    _assert_prediction_schema(body)
    threshold = float(body["threshold"])
    if body["predicted_probability"] >= threshold:
        assert body["predicted_label"] == 1
        assert body["flagged_for_review"] is True
        assert body["risk_band"] != "Low Risk"


def test_invalid_category_returns_json_error(client) -> None:
    payload = dict(LOW_RISK_PAYLOAD)
    payload["Merchant_Category"] = "NotAMerchant"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    body = response.get_json()
    assert body == {"error": "Invalid merchant category"}
    assert "Traceback" not in response.get_data(as_text=True)


def test_missing_field_returns_json_error(client) -> None:
    payload = dict(LOW_RISK_PAYLOAD)
    payload.pop("Transaction_Amount")
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Missing required field: Transaction_Amount"


def test_negative_amount_returns_json_error(client) -> None:
    payload = dict(LOW_RISK_PAYLOAD)
    payload["Transaction_Amount"] = -25.0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid transaction amount"}


def test_nan_and_infinity_return_json_error(client) -> None:
    nan_payload = dict(LOW_RISK_PAYLOAD)
    nan_payload["Transaction_Amount"] = "NaN"
    nan_response = client.post("/api/predict", json=nan_payload)
    assert nan_response.status_code == 400
    assert nan_response.get_json() == {"error": "Invalid transaction amount"}

    inf_payload = dict(LOW_RISK_PAYLOAD)
    inf_payload["Average_Spend"] = "Infinity"
    inf_response = client.post("/api/predict", json=inf_payload)
    assert inf_response.status_code == 400
    assert inf_response.get_json()["error"] == "Invalid average spend"

    bad_date = dict(LOW_RISK_PAYLOAD)
    bad_date["Transaction_Date"] = "2023/13/99 99:99"
    date_response = client.post("/api/predict", json=bad_date)
    assert date_response.status_code == 400
    assert date_response.get_json() == {"error": "Invalid transaction date"}


def test_threshold_behavior(client) -> None:
    health = client.get("/api/health").get_json()
    threshold = float(health["threshold"])
    assert threshold == pytest.approx(0.10)
    for payload in (LOW_RISK_PAYLOAD, HIGH_RISK_PAYLOAD):
        body = client.post("/api/predict", json=payload).get_json()
        assert float(body["threshold"]) == pytest.approx(threshold)
        flagged = body["predicted_probability"] >= threshold
        assert body["flagged_for_review"] is flagged
        assert body["predicted_label"] == int(flagged)


def test_risk_score_matches_probability(client) -> None:
    body = client.post("/api/predict", json=LOW_RISK_PAYLOAD).get_json()
    expected = float(body["predicted_probability"]) * 100.0
    assert float(body["risk_score"]) == pytest.approx(expected, abs=1e-6)


def test_risk_band_and_deterministic_prediction(client) -> None:
    first = client.post("/api/predict", json=HIGH_RISK_PAYLOAD)
    second = client.post("/api/predict", json=HIGH_RISK_PAYLOAD)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json() == second.get_json()
    band = first.get_json()["risk_band"]
    assert band in VALID_RISK_BAND_NAMES


def test_forbidden_fields_are_ignored(client) -> None:
    payload = dict(LOW_RISK_PAYLOAD)
    payload["Customer_ID"] = "CUST-HIDDEN"
    payload["Transaction_ID"] = "TXN-HIDDEN"
    payload["Fraudulent"] = 1
    payload["Suspicious_Keyword"] = "Yes"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "CUST-HIDDEN" not in text
    assert "TXN-HIDDEN" not in text


def test_frontend_is_served_and_uses_relative_api(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Financial Fraud Detection" in html
    assert "Machine Learning Based Fraud Risk Assessment" in html
    assert "localhost" not in html
    css = client.get("/styles.css")
    js = client.get("/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "/api/predict" in js.get_data(as_text=True)
    assert "127.0.0.1" not in js.get_data(as_text=True)
