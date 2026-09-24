# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

> **Author:** Hi, I'm Surag. This is my end-to-end Financial Fraud Detection & Risk Analytics system.
> 
> 🌐 **Live Web Application:** [https://financial-fraud-detection-sigma.vercel.app](https://financial-fraud-detection-sigma.vercel.app)

Internship / prototype project (**Project Foresight**). This is a machine-learning **fraud risk** system. It is not a certified banking production deployment and not a guarantee of fraud detection.

## Overview

The system scores financial transactions for estimated fraud risk using a frozen **Logistic Regression + Class Weight** pipeline, **sigmoid** probability calibration, and a **0.10** review threshold. A Streamlit dashboard supports historical analytics and single-transaction scoring.

The model does **not** prove that a transaction is fraudulent. A probability at or above 0.10 means **flag for human review**.

## Objectives

- Work on a highly imbalanced dataset (9.64% fraud) without treating accuracy as the goal.
- Engineer leakage-safe, row-local features shared by training and serving.
- Compare eight candidate models on an untouched test fold.
- Calibrate probabilities and choose the operating threshold on **training data only**.
- Present risk scores, risk bands, and review flags in a dashboard that does not retrain.
- Keep `Customer_ID`, `Transaction_ID`, `Suspicious_Keyword`, and `Fraudulent` out of model inputs.

## Dataset

Primary file: `data/raw/financial_fraud_detection_dataset.csv`

| Item | Value |
| --- | --- |
| Transactions | 5,000 |
| Columns | 14 |
| Fraud | 482 (9.64%) |
| Legitimate | 4,518 (90.36%) |
| Target | `Fraudulent` |
| Date range | 2023-01-01 to 2024-02-21 |
| Train / test | 4,000 / 1,000 (`test_size=0.20`, `stratify`, `random_state=42`) |
| Raw MD5 | `9a4a90ce2e07a717b4289dc95a71663c` |

Original internship files under `requirements_files/` are not used by the application and must not be modified.

## Architecture

```
Raw Transaction Data
        ↓
Data Validation
        ↓
EDA
        ↓
Feature Engineering (18 columns)
        ↓
Preprocessing (train-fit only)
        ↓
Candidate ML Models
        ↓
Model Evaluation
        ↓
Logistic Regression + Class Weight
        ↓
Probability Calibration (sigmoid)
        ↓
Production Threshold = 0.10
        ↓
Fraud Probability
        ↓
Risk Score (probability × 100)
        ↓
Risk Band
        ↓
Streamlit Dashboard
```

Leakage safeguards: split before fitting; no test-set preprocess/calibration/threshold; no IDs, keyword, or target in X; no full-file customer aggregates; dashboard inference only.

## Features

**Included (18):** `Transaction_Amount`, `Average_Spend`, `Previous_Transactions`, `Account_Age_Days`, `amount_to_average_ratio`, hour / day-of-week / month / year, `hour_sin`/`hour_cos`, `dow_sin`/`dow_cos`, `Is_International`, `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`.

**Excluded from X:** `Transaction_ID`, `Customer_ID`, `Suspicious_Keyword`, `Fraudulent`, raw `Transaction_Date`.

Preprocess (train-fit only): `RobustScaler` + `OneHotEncoder(handle_unknown="ignore")`.

## Models

Eight Phase 5/6 candidates: Logistic Regression, Decision Tree, Random Forest, XGBoost × {SMOTE, Class Weight}.

Accuracy was **not** used to pick a winner. XGBoost + SMOTE reached 0.8820 accuracy with only 0.0938 recall.

## Final model

- Logistic Regression + Class Weight
- Sigmoid `CalibratedClassifierCV` (5-fold, `ensemble=False`)
- Production threshold **0.10**
- Artifact: `models/production/final_fraud_pipeline.joblib`
- Production MD5: `5d08c124ddbb452f7a93f6982d02240a`

Risk score = predicted probability × 100.

| Score | Band |
| --- | --- |
| 0–9 | Low Risk |
| 10–19 | Medium Risk |
| 20–39 | High Risk |
| 40–100 | Critical Risk |

## Metrics

**Official Phase 7 test metrics** (untouched 1,000-row test set, 96 frauds, threshold 0.10):

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

On the untouched test set, the final model identified 73 of 96 fraudulent transactions while incorrectly flagging 234 legitimate transactions.

Phase 6 default-threshold scores (for example Accuracy 0.7480, F1 0.3668, FP 229) are **historical comparison only**.

## Dashboard

Five pages: Overview, Fraud Analytics, Model Performance, Transaction Prediction, Risk Monitoring.

```powershell
streamlit run dashboard/app.py
```

The dashboard loads the frozen joblib. It does not retrain, calibrate, or change the threshold.

## Installation

```powershell
cd C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

See `reports/setup_guide.md` for Windows troubleshooting.

## Run instructions

```powershell
python run_validation.py
streamlit run dashboard/app.py
```

Do not run training or Phase 7 freeze scripts to “improve” viva numbers.

## Web demo (Vercel)

🌐 **Live Production Deployment:** [https://financial-fraud-detection-sigma.vercel.app](https://financial-fraud-detection-sigma.vercel.app)

The frozen pipeline is also served as a same-origin Flask API plus a static form. Streamlit is **not** used on Vercel.

```powershell
python api/index.py
```

Then open `http://127.0.0.1:8000`. Deploy notes: `reports/vercel_deployment.md`.

```powershell
npm install -g vercel
vercel
vercel --prod
```

## Testing

```powershell
python -m pytest -q
```

Phase 9 freeze had 110 tests. The suite now includes the Vercel API tests as well. All should pass.

## Limitations

- Precision 23.78%: most flags are legitimate and need review.
- 23 of 96 test frauds are missed.
- Only 96 test positives: metric variance is real.
- Accuracy is a poor headline metric at 9.64% prevalence.
- Not a production banking system; `production_ready_claim` is false.

## Future enhancements

None of these are implemented: training-only customer history, keyword use only with proven provenance, cost-sensitive threshold search still on train/OOF, analyst case management. Do not retune on the current test fold.

## Project structure

| Path | Contents |
| --- | --- |
| `src/` | Validation, EDA, features, training, evaluation, inference |
| `api/` | Flask inference API for Vercel |
| `web/` | Public prediction frontend |
| `dashboard/` | Streamlit UI (local) |
| `data/raw/` | Frozen primary CSV |
| `models/production/` | Frozen serving artifact |
| `reports/` | Metrics tables and Phase 10 documentation |
| `artifacts/` | Charts; optional screenshots |
| `tests/` | pytest suite |

Full tree: `reports/final_project_structure.md`.

## Documentation

- `reports/final_project_documentation.md`
- `reports/viva_questions_answers.md`
- `reports/viva_2_minute_explanation.md`
- `reports/presentation_script.md`
- `reports/screenshot_plan.md`
- `reports/setup_guide.md`
- `reports/vercel_deployment.md`
