"""Vercel-compatible Flask inference API.

Loads the frozen Phase 7 pipeline once per process. Does not retrain, fit,
calibrate, or choose a new threshold. The Streamlit dashboard is unchanged.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from src.data.loader import DATE_FORMAT, ValidationError
from src.features.feature_engineering import FEATURE_INPUT_COLUMNS
from src.models.predict import (
    PRODUCTION_METADATA_PATH,
    load_production_pipeline,
    predict_transaction,
)

WEB_DIR = ROOT / "web"

ALLOWED_CATEGORIES: dict[str, frozenset[str]] = {
    "Merchant_Category": frozenset(
        {
            "Electronics",
            "Entertainment",
            "Fashion",
            "Food",
            "Grocery",
            "Health",
            "Travel",
            "Utilities",
        }
    ),
    "Payment_Method": frozenset(
        {"Credit Card", "Debit Card", "NetBanking", "PayPal", "UPI"}
    ),
    "Device_Type": frozenset({"Desktop", "Mobile", "POS"}),
    "Location": frozenset(
        {
            "Bengaluru",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Kolkata",
            "Mumbai",
            "Pune",
        }
    ),
}

NUMERIC_FIELDS = (
    "Transaction_Amount",
    "Average_Spend",
    "Previous_Transactions",
    "Account_Age_Days",
)
FORBIDDEN_INPUTS = (
    "Customer_ID",
    "Transaction_ID",
    "Fraudulent",
    "Suspicious_Keyword",
)
FIELD_LABELS = {
    "Transaction_Amount": "transaction amount",
    "Average_Spend": "average spend",
    "Previous_Transactions": "previous transactions",
    "Account_Age_Days": "account age",
    "Is_International": "international flag",
    "Transaction_Date": "transaction date",
    "Merchant_Category": "merchant category",
    "Payment_Method": "payment method",
    "Device_Type": "device type",
    "Location": "location",
}

app = Flask(__name__)


class ApiError(Exception):
    """Public JSON error. Never include a traceback."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@lru_cache(maxsize=1)
def _load_metadata() -> dict[str, Any]:
    if not PRODUCTION_METADATA_PATH.exists():
        raise ApiError("Production model is unavailable.", 503)
    payload = json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))
    threshold = float(payload.get("production_threshold", -1))
    if not 0.0 < threshold < 1.0:
        raise ApiError("Production model is unavailable.", 503)
    return payload


@lru_cache(maxsize=1)
def _load_pipeline():
    try:
        return load_production_pipeline()
    except FileNotFoundError as exc:
        raise ApiError("Production model is unavailable.", 503) from exc


def _public_threshold() -> float:
    return float(_load_metadata()["production_threshold"])


def _public_model_name() -> str:
    meta = _load_metadata()
    name = str(meta.get("model_name", "Logistic Regression")).strip()
    strategy = str(meta.get("strategy", "Class Weight")).strip()
    return f"{name} + {strategy}"


def _public_calibration() -> str:
    return str(_load_metadata().get("calibration_method", "sigmoid")).strip().title()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_finite_number(value: Any, field: str) -> float:
    label = FIELD_LABELS.get(field, field)
    if isinstance(value, bool):
        raise ApiError(f"Invalid {label}", 400)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            if field == "Transaction_Amount":
                raise ApiError("Invalid transaction amount", 400)
            raise ApiError(f"Invalid {label}", 400)
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        if field == "Transaction_Amount":
            raise ApiError("Invalid transaction amount", 400) from exc
        raise ApiError(f"Invalid {label}", 400) from exc
    if not math.isfinite(number):
        if field == "Transaction_Amount":
            raise ApiError("Invalid transaction amount", 400)
        raise ApiError(f"Invalid {label}", 400)
    return number


def _parse_international(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "international": 1,
            "domestic": 0,
            "international (1)": 1,
            "domestic (0)": 0,
        }
        if text in mapping:
            return mapping[text]
    number = _parse_finite_number(value, "Is_International")
    if number in (0, 1):
        return int(number)
    raise ApiError("Invalid international flag", 400)


def _parse_date(value: Any) -> str:
    if not isinstance(value, str):
        raise ApiError("Invalid transaction date", 400)
    text = value.strip()
    try:
        datetime.strptime(text, DATE_FORMAT)
    except ValueError as exc:
        raise ApiError("Invalid transaction date", 400) from exc
    return text


def _sanitize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object", 400)

    record: dict[str, Any] = {}
    for field in FEATURE_INPUT_COLUMNS:
        if field not in payload or _is_blank(payload.get(field)):
            raise ApiError(f"Missing required field: {field}", 400)
        record[field] = payload[field]

    for field in NUMERIC_FIELDS:
        number = _parse_finite_number(record[field], field)
        if number < 0:
            if field == "Transaction_Amount":
                raise ApiError("Invalid transaction amount", 400)
            raise ApiError(f"Invalid {FIELD_LABELS[field]}", 400)
        record[field] = number

    record["Is_International"] = _parse_international(record["Is_International"])
    record["Transaction_Date"] = _parse_date(record["Transaction_Date"])

    for field, allowed in ALLOWED_CATEGORIES.items():
        value = str(record[field]).strip()
        if value not in allowed:
            raise ApiError(f"Invalid {FIELD_LABELS[field]}", 400)
        record[field] = value

    return record


def _prediction_body(result: dict[str, Any]) -> dict[str, Any]:
    threshold = _public_threshold()
    probability = float(result["predicted_probability"])
    label = int(result["predicted_label"])
    flagged = bool(label == 1) or bool(probability >= threshold)
    return {
        "predicted_probability": probability,
        "risk_score": float(result["risk_score"]),
        "risk_band": str(result["risk_band"]),
        "predicted_label": label,
        "flagged_for_review": flagged,
        "threshold": threshold,
    }


def _json_error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return _json_error(exc.message, exc.status_code)


@app.errorhandler(ValidationError)
def _handle_validation_error(exc: ValidationError):
    text = str(exc)
    lowered = text.lower()
    if "transaction_amount" in lowered and (
        "negative" in lowered or "finite" in lowered or "nan" in lowered
    ):
        return _json_error("Invalid transaction amount", 400)
    if "transaction_date" in lowered:
        return _json_error("Invalid transaction date", 400)
    if "missing required" in lowered:
        return _json_error("Missing required field", 400)
    return _json_error("Invalid transaction", 400)


@app.errorhandler(404)
def _handle_not_found(_exc):
    return _json_error("Not found", 404)


@app.errorhandler(405)
def _handle_method(_exc):
    return _json_error("Method not allowed", 405)


@app.errorhandler(Exception)
def _handle_unexpected(exc):
    if isinstance(exc, ApiError):
        return _json_error(exc.message, exc.status_code)
    if isinstance(exc, ValidationError):
        return _handle_validation_error(exc)
    if isinstance(exc, HTTPException):
        return exc
    return _json_error("Unable to score this transaction", 500)


@app.get("/api/health")
@app.get("/health")
def health():
    body = {
        "status": "ok",
        "model": _public_model_name(),
        "threshold": _public_threshold(),
        "calibration": _public_calibration(),
    }
    return jsonify(body), 200


@app.post("/api/predict")
@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        raise ApiError("Request body must be a JSON object", 400)
    record = _sanitize_payload(payload)
    for forbidden in FORBIDDEN_INPUTS:
        record.pop(forbidden, None)
    result = predict_transaction(record, pipeline=_load_pipeline())
    return jsonify(_prediction_body(result)), 200


@app.get("/")
def frontend_index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/styles.css")
def frontend_styles():
    return send_from_directory(WEB_DIR, "styles.css")


@app.get("/app.js")
def frontend_script():
    return send_from_directory(WEB_DIR, "app.js")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
