# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

**Project Foresight — Final project documentation**  
Internship / academic prototype. Not a certified banking production system.

---

## 1. Title

Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard.

## 2. Abstract

Financial fraud is uncommon relative to legitimate payments, so a classifier that labels every transaction as legitimate can look accurate while catching no fraud. This project builds a leakage-safe machine-learning pipeline on `financial_fraud_detection_dataset.csv` (5,000 rows, 482 frauds, 9.64% fraud rate). Row-local feature engineering produces 18 model columns; identifiers, the target, raw timestamps, and `Suspicious_Keyword` are excluded from X. Eight candidate pipelines (Logistic Regression, Decision Tree, Random Forest, XGBoost × SMOTE or class weight) were compared on an untouched 1,000-row test fold. Logistic Regression with class weight was selected on F1, recall, precision, and interpretability—not accuracy. Sigmoid calibration on training data reduced inner-validation Brier score from 0.1567 to 0.0706. A production threshold of 0.10 was chosen from training out-of-fold probabilities to keep recall at least 0.70. The frozen model maps probability to a 0–100 risk score and four risk bands. A Streamlit dashboard scores new transactions without retraining. On the untouched test set the final operating point has recall 76.04% and precision 23.78% (73 of 96 frauds caught; 234 legitimate transactions flagged). These figures describe a review-support prototype, not guaranteed fraud detection.

## 3. Introduction

Payment systems need a way to rank transactions for analyst review. This project implements a reproducible Python pipeline (pandas, scikit-learn, imbalanced-learn, XGBoost, Plotly, Streamlit) with a strict split: train on 4,000 rows, evaluate once on 1,000 rows (`random_state=42`). Phases 1–9 covered dataset choice, validation, EDA, features, training, comparison, calibration, dashboard, and QA. Phase 10 documents the finished system for viva and presentation.

## 4. Problem Statement

Given a historical table of transactions with a binary `Fraudulent` label, estimate fraud **risk** for a new transaction and flag cases above a frozen threshold for human review. The positive class is rare (9.64%). False negatives miss fraud; false positives create review workload. Accuracy alone is not an acceptable objective.

## 5. Objectives

- Validate a single primary CSV without modifying it.
- Engineer leakage-safe features shared by training and serving.
- Compare SMOTE and class-weight strategies fairly on one test fold.
- Calibrate probabilities and choose a threshold on training data only.
- Ship a frozen inference artifact and a Streamlit application layer.
- Report honest test metrics, including false positives and false negatives.

## 6. Existing System / Traditional Approach

Rule-based checks (amount cut-offs, country lists, keyword flags) are easy to explain but brittle. A constant “always legitimate” rule on this test set is already 90.40% accurate with 0% recall. Amount-only rules are weak here: fraud and legitimate amount means are 81.99 vs 78.88. Keyword rules were not used as a model feature because timing/provenance is undocumented.

## 7. Proposed System

A supervised binary classifier with:

1. Schema validation (`run_validation.py`).
2. `build_features()` for 18 columns.
3. Train-only `RobustScaler` + `OneHotEncoder`.
4. Candidate models; selected Logistic Regression + class weight.
5. Sigmoid `CalibratedClassifierCV` (5-fold, `ensemble=False`).
6. Threshold 0.10 from training out-of-fold probabilities.
7. Risk score = probability × 100 and four bands.
8. Streamlit dashboard calling `predict_transaction()` / `predict_records()`.

## 8. Dataset Description

| Item | Value |
| --- | --- |
| File | `data/raw/financial_fraud_detection_dataset.csv` |
| Rows | 5,000 |
| Columns | 14 |
| Fraud | 482 (9.64%) |
| Legitimate | 4,518 (90.36%) |
| Missing | 0 |
| Duplicate Transaction_ID | 0 |
| Date range | 2023-01-01 02:16:00 to 2024-02-21 15:34:00 |
| Target | `Fraudulent` (0/1) |
| Raw MD5 | `9a4a90ce2e07a717b4289dc95a71663c` |

Columns: Transaction_ID, Customer_ID, Transaction_Date, Transaction_Amount, Merchant_Category, Payment_Method, Device_Type, Location, Is_International, Previous_Transactions, Average_Spend, Account_Age_Days, Suspicious_Keyword, Fraudulent.

Other CSVs and Power BI files under `requirements_files/` are reference material only.

Train/test split: 4,000 / 1,000; 386 / 96 fraud; rates 9.65% / 9.60%; no overlapping Transaction_IDs.

## 9. Data Preprocessing

- No row dropping, no outlier deletion (fraud can sit in the amount tail).
- Dates parsed with `%d-%m-%Y %H:%M`; invalid dates raise.
- After the split, a `ColumnTransformer` fits **on training rows only**: RobustScaler on numeric fields, OneHotEncoder with `handle_unknown="ignore"` and dense output.
- SMOTE, when used, is applied to transformed **train** data only via `imblearn.pipeline`.
- The test fold is never resampled and never used to fit scalers, encoders, calibration, or the threshold.

## 10. Exploratory Data Analysis

Verified observations from `reports/eda_summary.md`:

- Always-legitimate accuracy on the full file would be about 90.36%.
- International fraud rate 36.96% (163/441) vs domestic 7.00% (319/4559).
- Overnight hours 00:00–05:00 show higher observed fraud rates than 07:00–23:00.
- Amount correlation with the target is 0.0116; amount alone is a weak separator.
- `Suspicious_Keyword=Yes` has 47.31% fraud vs 7.57% when No; excluded from X pending provenance.
- Unique customers 3,847; median one transaction per customer.

## 11. Feature Engineering

`src/features/feature_engineering.py` is the single source of truth for training and the dashboard.

Engineered: `amount_to_average_ratio` (0 if Average_Spend ≤ 0), hour, day-of-week, month, year, `hour_sin/cos`, `dow_sin/cos`. Day-of-month is computed only for debug and is not in `MODEL_FEATURES`.

Cyclical encoding: `sin(2π · value / period)` and `cos(...)` for hour (period 24) and weekday (period 7) so 23:00 is close to 00:00.

Features are row-local: no full-file group-by, no target encoding, no customer fraud rate.

## 12. Machine Learning Methodology

- Stratified split, `test_size=0.20`, `random_state=42`.
- Four estimators × two imbalance strategies = eight pipelines.
- Class weight: `class_weight="balanced"` (sklearn) or `scale_pos_weight` from the **train** fold for XGBoost.
- Metrics: accuracy, precision, recall, F1 (`average="binary"`), ROC-AUC and PR-AUC from probabilities.
- Phase 6 comparison uses default 0.50 predictions for class metrics; Phase 7 changes the operating threshold after calibration using train-only OOF scores.

## 13. Model Comparison

Phase 6 untouched-test results (default threshold 0.50). Source: `reports/model_comparison.csv`.

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | SMOTE | 0.7470 | 0.2357 | 0.7292 | 0.3562 | 0.7682 | 0.2681 |
| Decision Tree | SMOTE | 0.7650 | 0.2186 | 0.5625 | 0.3149 | 0.7576 | 0.2335 |
| Random Forest | SMOTE | 0.7880 | 0.2434 | 0.5729 | 0.3416 | 0.7588 | 0.2235 |
| XGBoost | SMOTE | 0.8820 | 0.2250 | 0.0938 | 0.1324 | 0.7584 | 0.2336 |
| Logistic Regression | Class Weight | 0.7480 | 0.2417 | 0.7604 | 0.3668 | 0.7646 | 0.2620 |
| Decision Tree | Class Weight | 0.7530 | 0.2351 | 0.6979 | 0.3517 | 0.7639 | 0.2559 |
| Random Forest | Class Weight | 0.7360 | 0.2325 | 0.7604 | 0.3561 | 0.7577 | 0.2412 |
| XGBoost | Class Weight | 0.7830 | 0.2531 | 0.6458 | 0.3636 | 0.7509 | 0.2262 |

Dummy Always Legitimate (test set, not a trained model): Accuracy 0.9040, Recall 0.0000.

XGBoost SMOTE has the highest accuracy (0.8820) but recall 0.0938 (9 of 96 frauds). Accuracy was rejected as the selection metric.

## 14. Final Model Selection

**Selected candidate (Phase 6):** Logistic Regression + Class Weight.

Reasons from the actual table: highest F1 (0.3668), tied-highest recall (0.7604) with Random Forest Class Weight, better precision (0.2417 vs 0.2325), simpler than trees/boosting. SMOTE Logistic Regression had slightly higher PR-AUC (0.2681 vs 0.2620) but lower F1/recall.

**Production artifact (Phase 7):** the same family, wrapped with sigmoid calibration and threshold 0.10. Path: `models/production/final_fraud_pipeline.joblib`. MD5: `5d08c124ddbb452f7a93f6982d02240a`.

## 15. Probability Calibration

Class-weighted logistic regression overstated fraud probability (inner-val mean predicted 0.339 vs observed 0.096). `CalibratedClassifierCV(method="sigmoid", cv=5, ensemble=False)` was compared on an inner 3,200/800 split of **training** data. Sigmoid Brier 0.0706 vs uncalibrated 0.1567; log loss 0.2427 vs 0.4905. Isotonic was similar but not selected (only 77 inner-val frauds). Test labels were not used.

## 16. Threshold Selection

After calibration, 0.50 recall on training OOF collapsed. A grid 0.10–0.90 (step 0.05) was scored on 5-fold OOF training probabilities. Primary rule: recall ≥ 0.70 and FPR ≤ 0.35, then maximize F1. **0.10** was the only grid point meeting the recall floor (train OOF recall 0.8161, FPR 0.2438, F1 0.3982). 0.15 had slightly higher F1 but recall 0.6658. The test set was scored **once** after freeze.

## 17. Risk Score & Risk Bands

`risk_score = predicted_probability × 100` (clipped to 0–100).

| Band | Score | Meaning |
| --- | --- | --- |
| Low Risk | 0–9 | Below review threshold |
| Medium Risk | 10–19 | Flagged, lower calibrated probability |
| High Risk | 20–39 | Higher calibrated probability |
| Critical Risk | 40–100 | Highest calibrated probability |

Flag if probability ≥ 0.10. The score is a **model risk indicator**, not proof of fraud.

Official untouched-test risk bands (`reports/risk_band_summary.csv`, 1,000 rows):

| Band | Score | Count | Historical frauds in band |
| --- | --- | --- | --- |
| Low Risk | 0–9 | 693 | 23 |
| Medium Risk | 10–19 | 150 | 35 |
| High Risk | 20–39 | 126 | 26 |
| Critical Risk | 40–100 | 31 | 12 |

Official test flagged 307/1,000 (30.70%). The 23 frauds in Low Risk are the false negatives. Dashboard Risk Monitoring scores the full 5,000-row historical file for analytics; those counts are not the official test table.

## 18. System Architecture

```
Raw transaction CSV
        ↓
Data validation (schema, missing, IDs, binary target)
        ↓
EDA (read-only copies)
        ↓
Feature engineering (18 columns, row-local)
        ↓
Stratified split (train 4000 / test 1000)
        ↓
Preprocess fit on train (RobustScaler + OneHotEncoder)
        ↓
Eight candidate ML pipelines
        ↓
Fair test evaluation (Phase 6)
        ↓
Logistic Regression + Class Weight
        ↓
Sigmoid calibration (train only)
        ↓
Production threshold = 0.10 (train OOF only)
        ↓
Fraud probability
        ↓
Risk score (× 100) and risk band
        ↓
Streamlit dashboard (inference only)
```

Leakage safeguards: split first; no test fitting; no target in X; no ID or keyword in X; no full-file customer aggregates; dashboard does not call `.fit(`.

## 19. Streamlit Dashboard

Launch (Windows PowerShell, venv active):

```powershell
streamlit run dashboard/app.py
```

The app hash-checks the raw CSV, loads `models/production/final_fraud_pipeline.joblib`, and reads official metrics from `reports/final_model_metrics.csv`. It never calls `.fit(`, never changes 0.10, and never collects Customer_ID, Transaction_ID, Suspicious_Keyword, or Fraudulent on the prediction form.

Sidebar: model name, sigmoid / 5-fold CV, threshold 0.10, ROC-AUC, recall, and the disclaimer that risk score is not proof of fraud.

### Overview

- **Purpose:** Land the visitor on dataset scale and the “risk, not proof” message.
- **KPIs:** total 5,000; fraud 482; legitimate 4,518; fraud rate 9.64%; count and percentage flagged on the **full historical file** (not the official test table).
- **Charts:** fraud vs legitimate pie; amount overlay histogram; merchant-category fraud rate; domestic vs international rate.
- **Interaction:** none beyond navigation.
- **Interpretation:** flagged KPIs are frozen-model scores on all 5,000 rows. Official precision/recall live on Model Performance.

### Fraud Analytics

- **Purpose:** Historical EDA with filters. Uses ground-truth `Fraudulent`, which is never sent to the model.
- **KPIs:** filtered row count vs 5,000.
- **Charts:** fraud rates by merchant, payment, device, location, domestic/international, hour; amount histogram.
- **Interaction:** multiselect filters. Empty filter = all rows. Tiny groups produce noisy rates.
- **Interpretation:** associations in this file (for example higher international rates), not causal proof.

### Model Performance

- **Purpose:** Show the frozen Phase 7 one-shot test evaluation.
- **KPIs:** Accuracy 74.30%, Precision 23.78%, Recall 76.04%, F1 36.23%, ROC-AUC 0.7646, PR-AUC 0.2620.
- **Charts:** confusion heatmap (TN 670, FP 234, FN 23, TP 73); precision/recall/F1 bars; ROC/PR bars; Always Legitimate (90.40% accuracy, 0% recall) vs production.
- **Interaction:** none. Numbers are not recomputed live.
- **Interpretation:** accuracy is not the success metric; 234 false positives are review cost; 23 false negatives are missed fraud.

### Transaction Prediction

- **Purpose:** Score one new transaction with the frozen pipeline.
- **Displayed:** probability, risk score, risk band, flag vs legitimate, threshold 0.10.
- **Charts:** result panel after submit.
- **Interaction:** form (amount, spend, previous count, account age, date/time, merchant, payment, device, location, domestic/international). Submit calls `predict_transaction()`. Invalid values show a short error.
- **Interpretation:** probability ≥ 0.10 means **flag for review**, not proven fraud. Low Risk is below the cut.

### Risk Monitoring

- **Purpose:** Rank historical rows by model risk for demo analytics.
- **KPIs:** counts in Low / Medium / High / Critical and flagged share on the historical file.
- **Charts:** risk-band bar; probability histogram; risk-score histogram; flagged vs not; band vs historical label.
- **Interaction:** filters and **Download scored results CSV** (will not overwrite `data/raw/`).
- **Interpretation:** historical labels are for comparison only. Full-file counts are not a substitute for the 1,000-row test metrics.

## 20. Results

Official frozen test evaluation (threshold 0.10):

| Metric | Result |
| --- | --- |
| Accuracy | 0.7430 |
| Precision | 0.2378 |
| Recall | 0.7604 |
| F1 | 0.3623 |
| ROC-AUC | 0.7646 |
| PR-AUC | 0.2620 |
| Threshold | 0.10 |
| TP | 73 |
| FP | 234 |
| TN | 670 |
| FN | 23 |

On the untouched test set, the final model identified 73 of 96 fraudulent transactions while incorrectly flagging 234 legitimate transactions. Ranking metrics match Phase 6 because sigmoid is monotonic; the operating point shifted slightly (FP 229 → 234) after calibration and the 0.10 cut.

## 21. Confusion Matrix Analysis

- **TP 73:** frauds caught.
- **FN 23:** frauds labelled legitimate — missed loss.
- **FP 234:** legitimate payments sent to review — analyst workload (FPR 25.88% of 904 legit rows).
- **TN 670:** correctly cleared legitimate payments.

About one in four flags is actual fraud (precision 23.78%). That is a real cost, not hidden.

## 22. Limitations

- 96 test frauds: intervals around recall/precision are wide.
- Precision is modest; review capacity is required.
- Date window is short (into February 2024); seasonality is not claimed.
- Keyword and customer history were excluded by design.
- Unknown categories are ignored by one-hot encoding rather than rejected.
- Prototype / internship scope: no live bank integration, no Kafka/Spark, no claimed 100% accuracy.

## 23. Future Enhancements

Possible later work (not implemented): training-only customer history, documented keyword timing, cost-sensitive threshold search still on train/OOF only, analyst case management. Do not retune on the current test fold.

## 24. Testing & Validation

- `python run_validation.py` — 5,000 rows, 482 fraud, hash check.
- `pytest -q` — 110 passed, 0 failed, 0 skipped (Phase 9).
- Inference tests: missing fields, negatives, NaN, inf, bad dates, empty frames.
- Streamlit AppTest walks all five pages.
- Raw MD5 and production model MD5 are asserted frozen.

## 25. Conclusion

The project delivers a reproducible fraud-**risk** prototype: a calibrated logistic model, a documented 0.10 review threshold, honest test metrics (recall 76.04%, precision 23.78%), and a dashboard that does not retrain. It is suitable for internship demonstration and viva, not as a guarantee of fraud or a production banking platform.

## 26. References

- Project reports: `reports/eda_summary.md`, `reports/feature_engineering_summary.md`, `reports/model_comparison.md`, `reports/phase7_model_selection.md`, `reports/phase9_qa_report.md`.
- scikit-learn documentation: calibration, metrics, preprocessing.
- imbalanced-learn: SMOTE (train-only candidates).
- XGBoost; Streamlit; Plotly; pandas; NumPy; joblib.
- Dataset: `financial_fraud_detection_dataset.csv` (project primary file).
