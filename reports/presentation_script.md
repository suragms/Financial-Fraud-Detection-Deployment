# 5-minute presentation script — Project Foresight

Sixteen slides. Speak the “Say” text; put only the “On slide” bullets on the board. Total talk time about five minutes if you keep each slide to 15–20 seconds except Results and Dashboard.

Do not claim 100% accuracy, guaranteed fraud, or bank-grade production.

---

## Slide 1 — Title

**On slide**

- Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard
- Internship prototype · Project Foresight
- Logistic Regression + Class Weight · threshold 0.10

**Say**

Good morning. This project estimates **fraud risk** for financial transactions with a frozen machine-learning pipeline and a Streamlit dashboard. It is a review-support prototype, not a certified banking product.

---

## Slide 2 — Problem Statement

**On slide**

- Fraud is rare → accuracy looks easy
- Need to catch fraud without flooding reviewers
- Output: probability, risk score, flag for **human review**

**Say**

Most payments are legitimate. A system that always says “not fraud” looks accurate and catches nothing. We need a score that ranks risky transactions for analysts.

---

## Slide 3 — Objectives

**On slide**

- Leakage-safe features and split
- Compare eight candidate models
- Calibrate probabilities; freeze threshold on train only
- Dashboard for analytics and single-transaction scoring
- Honest test metrics

**Say**

The goals were a clean dataset pipeline, a fair model comparison, a calibrated operating point chosen without peeking at test labels, and a dashboard that does not retrain.

---

## Slide 4 — Dataset

**On slide**

- File: `financial_fraud_detection_dataset.csv`
- 5,000 rows · 14 columns
- 482 fraud · 4,518 legitimate · **9.64%** fraud
- Train 4,000 / Test 1,000 · seed 42
- Raw MD5 `9a4a90ce2e07a717b4289dc95a71663c`

**Say**

We used one primary CSV: five thousand transactions, four hundred eighty-two frauds. Twenty percent held out, stratified, random state forty-two.

---

## Slide 5 — Data Preprocessing

**On slide**

- No row dropping for outliers
- RobustScaler + OneHotEncoder (`unknown = ignore`)
- Fit on **train only**
- SMOTE only inside some train pipelines — never on test

**Say**

Scaling and encoding are fitted on training data only. We did not delete outliers, because unusual amounts can be fraud.

---

## Slide 6 — EDA

**On slide**

- International historical fraud **36.96%** vs domestic **7.00%**
- Amount correlation with fraud ≈ **0.0116** (weak)
- Overnight hours higher observed rates
- Keyword associated with fraud but **not used as a feature**

**Say**

Exploratory analysis showed international transactions and overnight hours look riskier in this file. Amount alone is a weak separator. The suspicious-keyword field looks strong but we excluded it to avoid possible label leakage.

---

## Slide 7 — Feature Engineering

**On slide**

- 18 model columns, row-local
- amount_to_average_ratio
- hour / weekday / month / year + **cyclical sin/cos**
- Excluded: Transaction_ID, Customer_ID, Suspicious_Keyword, Fraudulent, raw date

**Say**

Every feature is computed from that one row. Cyclical encoding keeps 11 p.m. close to midnight. Identifiers and the target never enter X.

---

## Slide 8 — Models

**On slide**

- Logistic Regression · Decision Tree · Random Forest · XGBoost
- × SMOTE **or** Class Weight → **8 pipelines**
- Dummy Always Legitimate as a sanity check

**Say**

We trained four algorithms with two imbalance strategies. We also scored a dummy that always predicts legitimate.

---

## Slide 9 — Model Comparison

**On slide** (Phase 6, default threshold 0.50)

| Model | Strategy | Acc | Prec | Rec | F1 | ROC | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR | SMOTE | 0.7470 | 0.2357 | 0.7292 | 0.3562 | 0.7682 | 0.2681 |
| DT | SMOTE | 0.7650 | 0.2186 | 0.5625 | 0.3149 | 0.7576 | 0.2335 |
| RF | SMOTE | 0.7880 | 0.2434 | 0.5729 | 0.3416 | 0.7588 | 0.2235 |
| XGB | SMOTE | **0.8820** | 0.2250 | **0.0938** | 0.1324 | 0.7584 | 0.2336 |
| LR | Class Weight | 0.7480 | 0.2417 | **0.7604** | **0.3668** | 0.7646 | 0.2620 |
| DT | Class Weight | 0.7530 | 0.2351 | 0.6979 | 0.3517 | 0.7639 | 0.2559 |
| RF | Class Weight | 0.7360 | 0.2325 | 0.7604 | 0.3561 | 0.7577 | 0.2412 |
| XGB | Class Weight | 0.7830 | 0.2531 | 0.6458 | 0.3636 | 0.7509 | 0.2262 |

Highlight: **XGBoost SMOTE accuracy is a trap. Selected: LR + Class Weight.**

**Say**

XGBoost with SMOTE has the best accuracy but recalls under ten percent of fraud. We rejected accuracy as the winner metric. Logistic regression with class weight had the best F1 and high recall.

---

## Slide 10 — Final Model

**On slide**

- Logistic Regression + Class Weight
- Sigmoid calibration, 5-fold, `ensemble=False`
- Threshold **0.10** from **training OOF** (recall ≥ 0.70, FPR ≤ 0.35)
- Artifact: `models/production/final_fraud_pipeline.joblib`
- Model MD5 `5d08c124ddbb452f7a93f6982d02240a`

**Say**

We wrapped the selected model with sigmoid calibration because uncalibrated probabilities were too high. After calibration, 0.50 no longer catches fraud, so we froze 0.10 from training folds only.

---

## Slide 11 — Results

**On slide** (official test, threshold 0.10)

| Metric | Result |
| --- | --- |
| Accuracy | 0.7430 |
| Precision | 0.2378 |
| Recall | 0.7604 |
| F1 | 0.3623 |
| ROC-AUC | 0.7646 |
| PR-AUC | 0.2620 |
| Threshold | 0.10 |
| TP / FP / TN / FN | 73 / 234 / 670 / 23 |

**Say**

On the untouched test set, the final model identified **73 of 96** fraudulent transactions while incorrectly flagging **234** legitimate transactions. Recall is about seventy-six percent; precision is about twenty-four percent. We do not hide the false positives.

---

## Slide 12 — Risk Scoring

**On slide**

- `risk_score = probability × 100`
- 0–9 Low · 10–19 Medium · 20–39 High · 40–100 Critical
- Flag if probability ≥ 0.10
- **Not proof of fraud**

**Say**

The score is the model’s estimated risk on a 0–100 scale. Low means below the review cut. A high score still needs an analyst.

---

## Slide 13 — Dashboard

**On slide**

- `streamlit run dashboard/app.py`
- Overview · Fraud Analytics · Model Performance · Transaction Prediction · Risk Monitoring
- Loads frozen joblib · **does not retrain**

**Say**

Five pages. Overview and analytics use historical labels for charts only. Prediction scores a form without IDs or the keyword. Monitoring shows risk bands and a CSV download that cannot overwrite the raw dataset.

---

## Slide 14 — Limitations

**On slide**

- Precision 23.78% → review workload
- 23 missed frauds on the test fold
- 96 test frauds → noisy metrics
- Not a live bank system; no 100% claim

**Say**

This is not production-grade banking software. Precision is modest, some fraud is missed, and the test positive count is small.

---

## Slide 15 — Future Scope

**On slide**

- Training-only customer history
- Keyword only if provenance is proven
- Cost-sensitive threshold still on train/OOF
- Analyst workflow (not built)

**Say**

Future work would stay leakage-safe. We would not retune on this test set.

---

## Slide 16 — Conclusion

**On slide**

- Leakage-safe fraud-**risk** prototype
- Final model: LR + class weight, sigmoid, threshold 0.10
- Test recall 76.04%, precision 23.78%
- Honest internship demonstration · thank you

**Say**

To close: we built a reproducible risk system with a frozen calibrated logistic model, a 0.10 review threshold, and a dashboard that does not retrain. Results are useful for ranking reviews, not for claiming perfect fraud detection. Thank you. Questions?
