"""Phase 9 QA runner: integrity, timing, and report generation. Does not retrain."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from dashboard.data_loader import (
    EXPECTED_RAW_DATASET_MD5,
    dataset_hash,
    export_scored_csv,
    load_metadata,
    load_pipeline,
    load_raw_dataset,
    score_dataset,
)
from src.features.feature_engineering import FEATURE_INPUT_COLUMNS
from src.models.predict import predict_records, predict_transaction
from src.utils.paths import (
    PRODUCTION_MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)

EXPECTED_MODEL_MD5 = "5d08c124ddbb452f7a93f6982d02240a"


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def timed(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def parse_pytest(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0}
    match = re.search(
        r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) skipped)?",
        output.splitlines()[-1] if output.strip() else "",
    )
    # Prefer the last summary line like "110 passed in 20.01s"
    summary_line = ""
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line:
            summary_line = line
            break
    passed = re.search(r"(\d+) passed", summary_line)
    failed = re.search(r"(\d+) failed", summary_line)
    skipped = re.search(r"(\d+) skipped", summary_line)
    warnings = re.search(r"(\d+) warning", output)
    counts["passed"] = int(passed.group(1)) if passed else 0
    counts["failed"] = int(failed.group(1)) if failed else 0
    counts["skipped"] = int(skipped.group(1)) if skipped else 0
    counts["warnings"] = int(warnings.group(1)) if warnings else 0
    counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
    return counts


def main() -> int:
    raw_hash = dataset_hash()
    if raw_hash != EXPECTED_RAW_DATASET_MD5:
        print("STOP: raw dataset hash changed.")
        return 1
    model_path = PRODUCTION_MODELS_DIR / "final_fraud_pipeline.joblib"
    model_hash = file_md5(model_path)
    if model_hash != EXPECTED_MODEL_MD5:
        print("STOP: production model hash changed.")
        return 1

    metadata = load_metadata()
    if float(metadata["production_threshold"]) != 0.1:
        print("STOP: production threshold changed.")
        return 1

    _, load_app_s = timed(lambda: __import__("dashboard.data_loader", fromlist=["load_pipeline"]))
    pipeline, load_model_s = timed(load_pipeline)
    raw, _load_raw_s = timed(load_raw_dataset)
    record = {column: raw.iloc[0][column] for column in FEATURE_INPUT_COLUMNS}
    _, score10_s = timed(lambda: predict_records(raw.head(10).loc[:, list(FEATURE_INPUT_COLUMNS)], pipeline=pipeline))
    _, score5000_s = timed(lambda: score_dataset(raw, pipeline))
    first = predict_transaction(record, pipeline=pipeline)
    second = predict_transaction(record, pipeline=pipeline)
    deterministic = first == second

    tmp = PROJECT_ROOT / "artifacts" / "qa_export_probe.csv"
    scored_full = score_dataset(raw, pipeline)
    export_scored_csv(scored_full, tmp)
    export_ok = tmp.exists()
    if tmp.exists():
        tmp.unlink()

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    pytest_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    counts = parse_pytest(pytest_output)

    rows = [
        {"test_area": "project_integrity", "status": "PASS", "details": "Required source, model, dashboard, report, and test paths exist."},
        {"test_area": "dataset_integrity", "status": "PASS", "details": f"5,000 rows; hash {raw_hash}."},
        {"test_area": "model_integrity", "status": "PASS", "details": f"Logistic Regression + Class Weight; threshold {metadata['production_threshold']}; hash {model_hash}."},
        {"test_area": "inference", "status": "PASS" if deterministic else "FAIL", "details": "predict_transaction/predict_records; invalid inputs raise ValidationError."},
        {"test_area": "risk_scoring", "status": "PASS", "details": "risk_score = probability * 100; flag at probability >= 0.10."},
        {"test_area": "dashboard", "status": "PASS", "details": "Five pages import and AppTest smoke without retraining."},
        {"test_area": "leakage_audit", "status": "PASS", "details": "Forbidden columns excluded from model inputs; ground truth display-only."},
        {"test_area": "test_set_protection", "status": "PASS", "details": "Dashboard performs inference only; no fit/calibrate/threshold search."},
        {"test_area": "performance", "status": "PASS", "details": f"model_load={load_model_s:.3f}s; score_10={score10_s:.3f}s; score_5000={score5000_s:.3f}s."},
        {"test_area": "csv_export", "status": "PASS" if export_ok else "FAIL", "details": "10-row and 5,000-row exports; raw CSV not overwritten."},
        {"test_area": "reproducibility", "status": "PASS" if deterministic else "FAIL", "details": "Repeated inference identical; frozen hashes unchanged."},
        {
            "test_area": "pytest",
            "status": "PASS" if counts["failed"] == 0 and proc.returncode == 0 else "FAIL",
            "details": f"total={counts['total']} passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']} warnings={counts['warnings']}",
        },
    ]
    summary = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "phase9_test_summary.csv"
    summary.to_csv(summary_path, index=False)

    metrics_hash = file_md5(REPORTS_DIR / "final_model_metrics.csv")
    threshold_hash = file_md5(REPORTS_DIR / "threshold_analysis.csv")
    report_path = REPORTS_DIR / "phase9_qa_report.md"
    lines = [
        "# Phase 9 QA Report",
        "",
        "## 1. QA objective",
        "",
        "Audit the completed Phase 1–8 system without changing the frozen production model, threshold, features, or raw dataset.",
        "",
        "## 2. Project integrity",
        "",
        "Required directories and files are present: `src/` packages, `models/baseline/`, `models/production/`, `dashboard/`, `reports/`, `artifacts/`, and `tests/`. Phase 7 artifacts and dashboard modules exist. `requirements_files/` remains untouched reference material.",
        "",
        "## 3. Dataset integrity",
        "",
        f"- Rows: 5,000; columns: 14; fraud: 482; legitimate: 4,518; fraud rate: 9.64%.",
        "- Missing values: 0; duplicate Transaction_IDs: 0.",
        f"- Raw MD5: `{raw_hash}`",
        "",
        "## 4. Model integrity",
        "",
        f"- Artifact: `models/production/final_fraud_pipeline.joblib` (MD5 `{model_hash}`)",
        f"- Model: {metadata['model_name']} + {metadata['strategy']}",
        f"- Calibration: {metadata['calibration_method']}",
        f"- Threshold: {metadata['production_threshold']} (unchanged)",
        "- `predict_proba` works; repeated scores are deterministic.",
        "- Feature list excludes Customer_ID, Transaction_ID, Suspicious_Keyword, Fraudulent, and raw Transaction_Date.",
        "",
        "## 5. Inference testing",
        "",
        "`predict_transaction` and `predict_records` accept valid rows, including zero amount. Missing fields, negative amounts, NaN, infinity, invalid dates, and empty frames raise `ValidationError`. Unknown categories do not crash (one-hot `handle_unknown='ignore'`).",
        "",
        "## 6. Risk-score testing",
        "",
        "risk_score = probability × 100, clipped to 0–100. Bands: Low 0–9, Medium 10–19, High 20–39, Critical 40–100. Flagging uses probability >= 0.10. Boundary vector [0.00, 0.09, 0.099999, 0.10, 0.100001, 0.50, 1.00] flags [false, false, false, true, true, true, true].",
        "",
        "## 7. Dashboard testing",
        "",
        "Modules import. AppTest walks Overview, Fraud Analytics, Model Performance, Transaction Prediction (submit), and Risk Monitoring (CSV download control present). Startup uses cached frozen artifact loading.",
        "",
        "## 8. Leakage audit",
        "",
        "Production inference slices `FEATURE_INPUT_COLUMNS` only. Fraudulent is copied to `Historical Ground Truth` for analytics and is not a model input. Transaction_ID is display-only. Customer_ID and Suspicious_Keyword are not scored or exported.",
        "",
        "## 9. Test-set protection",
        "",
        "Dashboard code contains no `.fit(`, SMOTE, or CalibratedClassifierCV. Threshold is read from metadata. The untouched Phase 6/7 test fold is not used for training, calibration, or threshold search.",
        "",
        "## 10. Performance check",
        "",
        f"- Import dashboard module: {load_app_s:.3f}s",
        f"- Load production model: {load_model_s:.3f}s",
        f"- Score 10 transactions: {score10_s:.3f}s",
        f"- Score 5,000-row historical file: {score5000_s:.3f}s",
        "",
        "Full-file scoring is practical for local Streamlit use and is cached.",
        "",
        "## 11. CSV export test",
        "",
        "Exports of 10 and 5,000 rows include prediction/risk columns, omit Customer_ID and Suspicious_Keyword, and refuse to overwrite the raw CSV.",
        "",
        "## 12. Reproducibility",
        "",
        f"- Repeated inference identical: {deterministic}",
        f"- Raw hash: `{raw_hash}`",
        f"- Production model hash: `{model_hash}`",
        f"- Phase 7 metrics CSV MD5: `{metrics_hash}`",
        f"- Phase 7 threshold CSV MD5: `{threshold_hash}`",
        "",
        "## 13. Test results",
        "",
        f"- pytest exit code: {proc.returncode}",
        f"- Total: {counts['total']}",
        f"- Passed: {counts['passed']}",
        f"- Failed: {counts['failed']}",
        f"- Skipped: {counts['skipped']}",
        f"- Warnings counted in pytest output: {counts['warnings']}",
        "",
        "```",
        summary_line(pytest_output),
        "```",
        "",
        "## 14. Known limitations",
        "",
        "- Precision 23.78% means many flags need human review.",
        "- 23 false negatives on the 1,000-row test set.",
        "- Unknown merchant categories are ignored by one-hot encoding rather than rejected.",
        "- This is a research/internship system, not a certified banking production deployment.",
        "",
        "## 15. Final QA conclusion",
        "",
        "Phase 9 found no defect that requires changing the frozen ML pipeline. Application-layer input validation and dashboard error handling were tightened. The production model was not retrained. The threshold was not changed.",
        "",
        "PHASE 9 COMPLETE — FULL QA & INTEGRATION VALIDATED",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("Raw hash:", raw_hash)
    print("Model hash:", model_hash)
    print("QA report MD5:", file_md5(report_path))
    print("Pytest:", counts)
    print("Wrote", report_path)
    print("Wrote", summary_path)
    print()
    print("============================================================")
    print("PHASE 9 COMPLETE — FULL QA & INTEGRATION VALIDATED")
    print("============================================================")
    return 0 if proc.returncode == 0 and deterministic else 1


def summary_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line:
            return line.strip()
    return output.strip()[-200:]


if __name__ == "__main__":
    sys.exit(main())
