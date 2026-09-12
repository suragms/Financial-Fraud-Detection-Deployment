# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

## Project objective

Detect fraudulent financial transactions with a leakage-safe machine-learning pipeline and present risk analytics through an interactive Streamlit dashboard.

## Current phase

**Phase 8 — Streamlit fraud detection dashboard.**

The production model, features, calibration, threshold, and raw dataset are frozen. The dashboard is an inference and analytics application only. It does not retrain.

## Dataset

Primary dataset: `data/raw/financial_fraud_detection_dataset.csv`

- 5,000 rows, 482 frauds (9.64%)
- Target column: `Fraudulent` (historical ground truth for analytics only)
- Application code loads this file through project-relative paths
- Original reference files under `requirements_files/` are not modified
- Raw MD5: `9a4a90ce2e07a717b4289dc95a71663c`

## How to install dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

## How to run the dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Dashboard pages

1. **Overview** — KPI cards, fraud vs legitimate mix, amount distribution, category and international rates.
2. **Fraud Analytics** — interactive historical fraud-rate charts with filters and sample sizes.
3. **Model Performance** — frozen Phase 7 test metrics, confusion matrix, accuracy vs Always Legitimate baseline.
4. **Transaction Prediction** — scores one transaction through `src.models.predict.predict_transaction`.
5. **Risk Monitoring** — scores the historical file with the frozen pipeline, risk-band charts, table filters, CSV download.

## Model information

- Model: Logistic Regression + Class Weight
- Calibration: sigmoid / 5-fold CV (`ensemble=False`)
- Production artifact: `models/production/final_fraud_pipeline.joblib`
- Production threshold: **0.10** (read from `models/production/model_metadata.json`)
- Frozen test recall: 0.7604
- Frozen test ROC-AUC: 0.7646
- Frozen test precision: 0.2378
- Frozen test F1: 0.3623

The dashboard never fits preprocessing, calibration, or a new threshold.

## Risk-score explanation

`risk_score = predicted_probability × 100` (0–100).

| Band | Score |
| --- | --- |
| Low Risk | 0–9 |
| Medium Risk | 10–19 |
| High Risk | 20–39 |
| Critical Risk | 40–100 |

A score at or above 10 (probability ≥ 0.10) is flagged for review. The risk score is a **model-generated risk indicator, not proof of fraud**. Flagged transactions should be reviewed.

## Limitations

- 96 frauds in the untouched test set: metric variance is real.
- Precision is 23.78%: many flags are legitimate and create review workload.
- 23 test frauds are missed (false negatives).
- `Suspicious_Keyword`, `Customer_ID`, and `Transaction_ID` are not model inputs.
- Accuracy is a poor headline metric on this 9.64% fraud-rate dataset.

## Screenshot placeholders

Add screenshots here after a local run:

- `docs/screenshots/overview.png`
- `docs/screenshots/fraud_analytics.png`
- `docs/screenshots/model_performance.png`
- `docs/screenshots/prediction.png`
- `docs/screenshots/risk_monitoring.png`

## Project structure

```
Financial_Fraud_Detection/
├── dashboard/                 # Phase 8 Streamlit app
│   ├── app.py
│   ├── components.py
│   ├── data_loader.py
│   └── styles.py
├── data/raw/                  # primary CSV (do not edit)
├── models/baseline/           # Phase 5 candidates
├── models/production/         # frozen Phase 7 artifact
├── notebooks/01_eda.ipynb
├── reports/                   # EDA, comparison, Phase 7 metrics
├── src/data/
├── src/features/
├── src/models/                # training, scoring, predict.py
├── src/evaluation/
├── tests/
├── run_validation.py
├── run_training.py
├── run_evaluation.py
├── run_phase7.py
└── requirements.txt
```

## How to validate the dataset

```bash
python run_validation.py
```

## How to run tests

```bash
pytest
```

## Other commands

```bash
python run_training.py      # Phase 5 candidates (already saved)
python run_evaluation.py    # Phase 6 comparison (already saved)
python run_phase7.py        # Phase 7 production freeze (already saved)
```

## Notes

- `RANDOM_STATE = 42` is the project seed.
- Do not modify `data/raw/financial_fraud_detection_dataset.csv`.
- Do not overwrite `models/production/final_fraud_pipeline.joblib` from the dashboard.
