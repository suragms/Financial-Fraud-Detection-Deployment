# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

> **Author:** Hi, I'm Surag. This is my end-to-end Financial Fraud Detection & Risk Analytics system.
> 
> 🌐 **Live Web Application (Vercel):** [https://financial-fraud-detection-eta.vercel.app/](https://financial-fraud-detection-eta.vercel.app/)
> 
> 📦 **GitHub Repository:** [https://github.com/suragms/Financial-Fraud-Detection-Deployment.git](https://github.com/suragms/Financial-Fraud-Detection-Deployment.git)

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

```bash
# Clone the repository
git clone https://github.com/suragms/Financial-Fraud-Detection-Deployment.git
cd Financial-Fraud-Detection-Deployment

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# For Vercel Flask API only:
pip install -r requirements.txt

# For Streamlit Dashboard:
pip install -r requirements-streamlit.txt

# For full local development and testing:
pip install -r requirements-dev.txt
```

## Local Execution

### Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to view the 5 dashboard pages:
- **Overview**: High-level KPIs and historical ground-truth fraud distributions.
- **Fraud Analytics**: Multidimensional filtering across merchant categories, payment methods, devices, and times.
- **Model Performance**: Test set confusion matrix, ROC-AUC, PR-AUC, and precision/recall trade-offs.
- **Transaction Prediction**: Interactive single-transaction scoring using the frozen pipeline.
- **Risk Monitoring**: Risk-band distribution (Low, Medium, High, Critical) and bulk batch scoring.

### Flask API & Web Frontend
```bash
python api/index.py
```
Open `http://127.0.0.1:8000` to interact with the static frontend or test API endpoints:
- `GET /api/health`
- `POST /api/predict`

## Deployment

### Architecture

```
GitHub (suragms/Financial-Fraud-Detection-Deployment)
   │
   ├───► Vercel (Free Serverless)
   │       ├── Static Frontend (web/index.html, styles.css, app.js)
   │       └── Flask REST API (api/index.py → /api/health, /api/predict)
   │
   └───► Streamlit Community Cloud (Free App Hosting)
           └── Analytics Dashboard (dashboard/app.py)
```

### Why Streamlit is Deployed Separately from Vercel

- **Vercel** specializes in lightning-fast static hosting and ephemeral serverless execution. Hosting Flask API endpoints (`/api/predict`, `/api/health`) and the static HTML/CSS/JS frontend on Vercel delivers sub-second cold starts and global CDN delivery within the Vercel free tier limits.
- **Streamlit Community Cloud** provides persistent, interactive Python application runtimes specifically designed for stateful dashboards, reactive Plotly charts, and exploratory risk monitoring.
- **Shared Production Artifacts**: Both deployment targets share the identical frozen pipeline (`models/production/final_fraud_pipeline.joblib`), metadata (`models/production/model_metadata.json`), and feature engineering logic (`src/features/feature_engineering.py`).
- **Completely Free**: No Render server, container service, or paid hosting is required.

### Streamlit Community Cloud Deployment Steps

1. **Push repository to GitHub**: Ensure the repository `https://github.com/suragms/Financial-Fraud-Detection-Deployment.git` is up-to-date on GitHub on branch `main`.
2. **Open Streamlit Community Cloud**: Navigate to [share.streamlit.io](https://share.streamlit.io/) in your browser.
3. **Sign in with GitHub**: Authorize Streamlit Community Cloud to access your repositories.
4. **Select Repository**:
   - Repository: `suragms/Financial-Fraud-Detection-Deployment`
   - Branch: `main`
5. **Set Main File Path**:
   - Main file path: `dashboard/app.py`
6. **Deploy**: Click **Deploy!**. Streamlit automatically detects `dashboard/requirements.txt` (or `requirements-streamlit.txt`) and installs the dependencies.
7. **Secrets & Environment Variables**: None required! The application, frozen model, and historical reference dataset are completely self-contained.
8. **Verify Dashboard**: Verify that the dashboard loads smoothly without stack traces across all five pages:
   - Overview
   - Fraud Analytics
   - Model Performance
   - Transaction Prediction
   - Risk Monitoring

### Vercel Deployment Steps

🌐 **Live Verified Deployment:** [https://financial-fraud-detection-eta.vercel.app/](https://financial-fraud-detection-eta.vercel.app/)

1. **Connect to Vercel**: Import the GitHub repository into your Vercel dashboard or run from the CLI:
   ```bash
   vercel --prod
   ```
2. **Configuration**: Vercel reads `vercel.json` and automatically builds `api/index.py` as a serverless function with `models/production/`, `src/`, and `web/` bundled.
3. **Verify Endpoints**:
   - `GET /` → Serves `web/index.html`
   - `GET /styles.css` → Serves stylesheet
   - `GET /app.js` → Serves client-side application logic
   - `GET /api/health` → Returns `{"calibration":"Sigmoid","model":"Logistic Regression + Class Weight","status":"ok","threshold":0.1}`
   - `POST /api/predict` → Returns scoring schema (`predicted_probability`, `risk_score`, `risk_band`, `predicted_label`, `flagged_for_review`, `threshold`)

### Sample Validation Transactions

#### Example 1: Low-Risk Transaction
```json
{
  "Transaction_Date": "15-06-2023 14:30",
  "Transaction_Amount": 50.0,
  "Average_Spend": 95.0,
  "Previous_Transactions": 20,
  "Account_Age_Days": 800,
  "Is_International": 0,
  "Merchant_Category": "Food",
  "Payment_Method": "Debit Card",
  "Device_Type": "Mobile",
  "Location": "Mumbai"
}
```
- **Probability:** ~0.019 (1.93%)
- **Risk Score:** ~1.93
- **Risk Band:** Low Risk
- **Predicted Label:** `0` (Legitimate)
- **Flagged for Review:** `false` (probability < 0.10)

#### Example 2: High-Risk Transaction
```json
{
  "Transaction_Date": "02-01-2023 02:15",
  "Transaction_Amount": 250.0,
  "Average_Spend": 40.0,
  "Previous_Transactions": 1,
  "Account_Age_Days": 15,
  "Is_International": 1,
  "Merchant_Category": "Travel",
  "Payment_Method": "Credit Card",
  "Device_Type": "POS",
  "Location": "Delhi"
}
```
- **Probability:** ~0.787 (78.73%)
- **Risk Score:** ~78.73
- **Risk Band:** Critical Risk
- **Predicted Label:** `1` (Fraud)
- **Flagged for Review:** `true` (probability ≥ 0.10)

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
