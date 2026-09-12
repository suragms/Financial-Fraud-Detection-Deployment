# Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard

## Project objective

Detect fraudulent financial transactions with machine learning and present risk analytics through an interactive Streamlit dashboard.

## Current phase

**Phase 3 — Exploratory data analysis.**

ML training and dashboard implementation are planned for later phases.

## Dataset

Primary dataset: `data/raw/financial_fraud_detection_dataset.csv`

- Target column: `Fraudulent`
- Application code loads this file through project-relative paths only
- Original reference files under `requirements_files/` are not modified

## Technology stack

Current stack:

- Python
- pandas
- NumPy
- Plotly
- pytest

Later phases may add scikit-learn, imbalanced-learn, XGBoost, Streamlit, and joblib.

## Project structure

```
Financial_Fraud_Detection/
├── requirements_files/      # original reference materials (do not edit)
├── data/raw/                # primary CSV used by the application
├── notebooks/01_eda.ipynb   # Phase 3 EDA
├── src/data/                # loading, validation, EDA helpers
├── src/utils/               # paths and random seed
├── reports/eda_summary.md
├── artifacts/eda/           # Plotly HTML charts from EDA
├── tests/
├── requirements.txt
├── README.md
└── run_validation.py
```

## How to validate the dataset

From the project root:

```bash
python run_validation.py
```

## How to run EDA

Open `notebooks/01_eda.ipynb` and run all cells from the project root (or from `notebooks/`; the first cell locates the project root).

The notebook writes:

- `reports/eda_summary.md`
- `artifacts/eda/*.html`

It does not modify `data/raw/financial_fraud_detection_dataset.csv`.

## How to run tests

```bash
pytest
```

## Notes

- `RANDOM_STATE = 42` in `src/utils/seeds.py` is the single seed for later modeling.
- `Suspicious_Keyword` is retained as a data column and is not used as an ML feature in this phase.
- No model performance is claimed. No models have been trained.
