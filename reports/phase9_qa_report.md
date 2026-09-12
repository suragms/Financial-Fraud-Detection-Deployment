# Phase 9 QA Report

## 1. QA objective

Audit the completed Phase 1–8 system without changing the frozen production model, threshold, features, or raw dataset.

## 2. Project integrity

Required directories and files are present: `src/` packages, `models/baseline/`, `models/production/`, `dashboard/`, `reports/`, `artifacts/`, and `tests/`. Phase 7 artifacts and dashboard modules exist. `requirements_files/` remains untouched reference material.

## 3. Dataset integrity

- Rows: 5,000; columns: 14; fraud: 482; legitimate: 4,518; fraud rate: 9.64%.
- Missing values: 0; duplicate Transaction_IDs: 0.
- Raw MD5: `9a4a90ce2e07a717b4289dc95a71663c`

## 4. Model integrity

- Artifact: `models/production/final_fraud_pipeline.joblib` (MD5 `5d08c124ddbb452f7a93f6982d02240a`)
- Model: Logistic Regression + Class Weight
- Calibration: sigmoid
- Threshold: 0.1 (unchanged)
- `predict_proba` works; repeated scores are deterministic.
- Feature list excludes Customer_ID, Transaction_ID, Suspicious_Keyword, Fraudulent, and raw Transaction_Date.

## 5. Inference testing

`predict_transaction` and `predict_records` accept valid rows, including zero amount. Missing fields, negative amounts, NaN, infinity, invalid dates, and empty frames raise `ValidationError`. Unknown categories do not crash (one-hot `handle_unknown='ignore'`).

## 6. Risk-score testing

risk_score = probability × 100, clipped to 0–100. Bands: Low 0–9, Medium 10–19, High 20–39, Critical 40–100. Flagging uses probability >= 0.10. Boundary vector [0.00, 0.09, 0.099999, 0.10, 0.100001, 0.50, 1.00] flags [false, false, false, true, true, true, true].

## 7. Dashboard testing

Modules import. AppTest walks Overview, Fraud Analytics, Model Performance, Transaction Prediction (submit), and Risk Monitoring (CSV download control present). Startup uses cached frozen artifact loading.

## 8. Leakage audit

Production inference slices `FEATURE_INPUT_COLUMNS` only. Fraudulent is copied to `Historical Ground Truth` for analytics and is not a model input. Transaction_ID is display-only. Customer_ID and Suspicious_Keyword are not scored or exported.

## 9. Test-set protection

Dashboard code contains no `.fit(`, SMOTE, or CalibratedClassifierCV. Threshold is read from metadata. The untouched Phase 6/7 test fold is not used for training, calibration, or threshold search.

## 10. Performance check

- Import dashboard module: 0.000s
- Load production model: 0.001s
- Score 10 transactions: 0.042s
- Score 5,000-row historical file: 0.031s

Full-file scoring is practical for local Streamlit use and is cached.

## 11. CSV export test

Exports of 10 and 5,000 rows include prediction/risk columns, omit Customer_ID and Suspicious_Keyword, and refuse to overwrite the raw CSV.

## 12. Reproducibility

- Repeated inference identical: True
- Raw hash: `9a4a90ce2e07a717b4289dc95a71663c`
- Production model hash: `5d08c124ddbb452f7a93f6982d02240a`
- Phase 7 metrics CSV MD5: `558624d04767ca2ab9a6af3633467e47`
- Phase 7 threshold CSV MD5: `3db01d1360d5e1d8ec084ec0a72cd7dd`

## 13. Test results

- pytest exit code: 0
- Total: 110
- Passed: 110
- Failed: 0
- Skipped: 0
- Warnings counted in pytest output: 0

```
110 passed in 8.29s
```

## 14. Known limitations

- Precision 23.78% means many flags need human review.
- 23 false negatives on the 1,000-row test set.
- Unknown merchant categories are ignored by one-hot encoding rather than rejected.
- This is a research/internship system, not a certified banking production deployment.

## 15. Final QA conclusion

Phase 9 found no defect that requires changing the frozen ML pipeline. Application-layer input validation and dashboard error handling were tightened. The production model was not retrained. The threshold was not changed.

PHASE 9 COMPLETE — FULL QA & INTEGRATION VALIDATED
