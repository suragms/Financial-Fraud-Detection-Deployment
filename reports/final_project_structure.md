# Final project structure — Project Foresight

Internship / prototype layout. Paths are relative to the repository root.

`requirements_files/` holds original internship hand-outs (extra CSVs, notebooks, Power BI). The **application does not read them**. Do not modify them.

`.venv/`, `.pytest_cache/`, and `.git/` are local/tooling and are not part of the academic deliverable.

---

## SOURCE CODE

Python package that implements validation, EDA, features, training, evaluation, calibration, and inference.

```
src/
  __init__.py
  data/
    loader.py          # load + schema validation of the primary CSV
    eda.py             # read-only exploratory summaries and charts
  features/
    feature_engineering.py   # 18 MODEL_FEATURES; single source of truth
  models/
    pipeline.py        # Phase 5 candidate training (SMOTE / class weight)
    phase7.py          # calibration, threshold, freeze production wrapper
    scoring.py         # FraudRiskPipeline, risk score, bands
    predict.py         # leakage-safe inference API for dashboard / CLI
  evaluation/
    metrics.py
    compare.py         # Phase 6 model comparison
  utils/
    paths.py           # all filesystem paths
    seeds.py           # RANDOM_STATE = 42
```

Root runners (thin entry points; they import `src/`):

```
run_validation.py
run_training.py
run_evaluation.py
run_phase7.py
run_phase9.py
```

Phase 10 does not require re-running training or Phase 7.

---

## DATA

```
data/
  raw/
    financial_fraud_detection_dataset.csv
  processed/                 # placeholder only; the app reads raw and engineers in memory
```

- 5,000 rows, 14 columns, 482 fraud
- Frozen MD5: `9a4a90ce2e07a717b4289dc95a71663c`
- Never overwritten by the dashboard CSV download

Reference-only (not used by the app):

```
requirements_files/
  financial_fraud_detection_dataset.csv
  other CSVs, notebooks, .pbix files
```

---

## MODELS

```
models/
  baseline/            # Phase 5/6 candidate joblibs + selected_model.json
  production/
    final_fraud_pipeline.joblib    # frozen serving artifact
    model_metadata.json            # threshold 0.10, bands, hashes of record
```

Production MD5: `5d08c124ddbb452f7a93f6982d02240a`  
Family: Logistic Regression + Class Weight + sigmoid calibration.

Do not replace these files during Phase 10.

---

## REPORTS

Markdown and CSV written by earlier phases plus this Phase 10 pack:

```
reports/
  eda_summary.md
  feature_engineering_summary.md
  training_log.md
  model_comparison.md
  model_comparison.csv
  phase7_model_selection.md
  final_model_metrics.csv
  threshold_analysis.csv
  risk_band_summary.csv
  phase9_qa_report.md
  phase9_test_summary.csv
  final_project_documentation.md
  screenshot_plan.md
  viva_questions_answers.md
  viva_2_minute_explanation.md
  presentation_script.md
  setup_guide.md
  final_project_structure.md
```

---

## ARTIFACTS

Generated charts and previews. Safe to regenerate EDA HTML; **do not** treat them as a second source of official metrics. Official numbers live in `reports/` and `models/production/model_metadata.json`.

```
artifacts/
  eda/                         # Plotly HTML + categorical summary CSV
  feature_preview.csv
  model_evaluation/            # Phase 6 ROC / PR / confusion HTML
  model_selection/             # Phase 7 calibration, threshold, bands HTML
  screenshots/                 # optional viva captures (see screenshot_plan.md)
```

---

## TESTS

```
tests/
  test_dataset_validation.py
  test_eda.py
  test_feature_engineering.py
  test_ml_pipeline.py
  test_model_evaluation.py
  test_phase7.py
  test_dashboard.py
  test_phase9.py
```

Run: `pytest -q`  
Phase 9 freeze: **110 passed**.

---

## DASHBOARD

Application layer only. Loads the frozen pipeline. Never fits.

```
dashboard/
  app.py              # five pages
  components.py       # Plotly charts, KPIs, prediction panel
  data_loader.py      # cache helpers, hash check, scoring of historical file
  styles.py
  __init__.py
```

Launch:

```powershell
streamlit run dashboard/app.py
```

Pages: Overview, Fraud Analytics, Model Performance, Transaction Prediction, Risk Monitoring.

---

## OTHER ROOT FILES

```
README.md
requirements.txt
.gitignore
notebooks/01_eda.ipynb
```

---

## What each major directory is for (one line)

| Area | Role |
| --- | --- |
| `src/` | Reproducible ML and inference code |
| `data/raw/` | Frozen primary dataset |
| `models/` | Trained candidates + frozen production artifact |
| `reports/` | Written evidence, metrics tables, viva docs |
| `artifacts/` | Charts and optional screenshots |
| `tests/` | Automated checks including hash and leakage guards |
| `dashboard/` | Streamlit UI over frozen inference |

---

## Leakage-related files to remember in a viva

- Features: `src/features/feature_engineering.py` (`EXCLUDED_FEATURES`)
- Split / preprocess: `src/models/pipeline.py`
- Threshold / calibration: `src/models/phase7.py` (train/OOF only)
- Serving: `src/models/predict.py` (strips forbidden columns)
- UI: `dashboard/app.py` (no `.fit(`)
