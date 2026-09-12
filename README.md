# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

## Project objective

Detect fraudulent financial transactions with machine learning and present risk analytics through an interactive Streamlit dashboard.

## Current phase

**Phase 2 — Project scaffold, data validation, and reproducible foundation.**

ML training and dashboard implementation are planned for later phases.

## Dataset

Primary dataset: `data/raw/financial_fraud_detection_dataset.csv`

- Target column: `Fraudulent`
- Application code loads this file through project-relative paths only
- Original reference files under `requirements_files/` are not modified

## Technology stack

Phase 2 uses:

- Python
- pandas
- NumPy
- pytest

Later phases may add scikit-learn, imbalanced-learn, XGBoost, Plotly, Streamlit, and joblib. Those packages are not required yet.

## Project structure

```
Financial_Fraud_Detection/
├── requirements_files/      # original reference materials (do not edit)
├── data/
│   ├── raw/                 # primary CSV used by the application
│   └── processed/           # later phases
├── notebooks/               # later EDA / experiments
├── src/
│   ├── data/                # loading and validation
│   ├── features/            # later feature engineering
│   ├── models/              # later training pipeline
│   ├── evaluation/          # later metrics
│   └── utils/               # paths and random seed
├── models/
├── reports/
├── artifacts/
├── dashboard/               # later Streamlit app
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── run_validation.py
```

## How to validate the dataset

From the project root:

```bash
python run_validation.py
```

The script loads the CSV, checks schema and data quality, prints a profile, and exits with a non-zero status if critical checks fail.

Expected success line:

```text
DATA VALIDATION PASSED
```

Row counts, fraud counts, and fraud rate are calculated from the file. They are not hard-coded.

## How to run tests

```bash
pytest
```

Tests check that the dataset exists, required columns and a binary target are present, transaction IDs are unique, and the fraud rate is between 0 and 1. They do not modify the dataset.

## Notes

- `RANDOM_STATE = 42` in `src/utils/seeds.py` is the single seed for later modeling.
- `Suspicious_Keyword` is retained as a data column and is not used as an ML feature in this phase.
- No model performance is claimed. No models have been trained.
>>>>>>> 0267d8a (Initial commit: Financial Fraud Detection project)
