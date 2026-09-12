# Screenshot capture plan — Project Foresight (Phase 10)

Do **not** insert placeholder or generated images into reports. Capture real UI only.

This repository does not include Playwright or Selenium. Automated dashboard screenshots are therefore **not** captured in this phase. Take the photos below by hand and store them under `artifacts/screenshots/` if you need them for a viva slide deck.

Recommended folder:

```
artifacts/screenshots/
  01_dashboard_overview.png
  02_fraud_analytics.png
  03_model_performance.png
  04_prediction_low_risk.png
  05_prediction_flagged.png
  06_risk_monitoring.png
  07_risk_band_distribution.png
  08_project_folder_structure.png
  09_pytest_results.png
  10_streamlit_launch.png
```

---

## Before you start

In Windows PowerShell, from the project root:

```powershell
cd C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Expected local URL: `http://localhost:8501`

The sidebar should show **Logistic Regression + Class Weight**, **Sigmoid / 5-fold CV**, **Threshold 0.10**.

If port 8501 is already in use, Streamlit will print an alternate URL. Use that URL instead.

---

## 1. Dashboard Overview

**File:** `01_dashboard_overview.png`

**Where:** sidebar → Overview.

**Capture:** full browser window including:

- title “Financial Fraud Detection System using Machine Learning”
- disclaimer that the system estimates risk and does not prove fraud
- KPIs: Total 5,000; Fraud 482; Legitimate 4,518; Fraud rate 9.64%; flagged count/percentage
- pie of historical fraud vs legitimate
- amount histogram
- merchant-category and domestic vs international rate charts

**Do not crop out the risk disclaimer.**

---

## 2. Fraud Analytics

**File:** `02_fraud_analytics.png`

**Where:** sidebar → Fraud Analytics.

**Capture:** filters at the top (leave empty for the full 5,000 rows) and the six rate charts plus the amount overlay histogram.

**Optional second shot:** filter **International** only so the higher international fraud rate is visible. Name that file `02b_fraud_analytics_international.png` if you take it.

---

## 3. Model Performance

**File:** `03_model_performance.png`

**Where:** sidebar → Model Performance.

**Must show:**

- Accuracy 74.30%, Precision 23.78%, Recall 76.04%, F1 36.23%, ROC-AUC 0.7646, PR-AUC 0.2620
- confusion matrix with TN 670, FP 234, FN 23, TP 73
- “Why accuracy is misleading” comparison vs Always Legitimate (90.40% accuracy, 0% recall)

These numbers come from the frozen Phase 7 CSV, not from a live recompute.

---

## 4. Transaction Prediction — Low Risk example

**File:** `04_prediction_low_risk.png`

**Where:** sidebar → Transaction Prediction.

Suggested **domestic** example that often scores below 0.10 (still capture whatever the frozen model actually returns; do not edit the numbers):

| Field | Suggested value |
| --- | --- |
| Transaction amount | 50 |
| Average spend | 95 |
| Previous transactions | 20 |
| Account age (days) | 800 |
| Date / time | 2023-06-15 14:30 |
| Merchant / payment / device / location | first list values |
| International | Domestic (0) |

Click **Score transaction**. Capture the result panel: probability, risk score, **Low Risk** (if that is what the model returns), and no review flag.

If this example is flagged, keep the screenshot anyway and label it as the actual model output. Do not photoshop the band.

---

## 5. Transaction Prediction — Flagged example

**File:** `05_prediction_flagged.png`

Suggested **international** example that is more likely to exceed 0.10:

| Field | Suggested value |
| --- | --- |
| Transaction amount | 250 |
| Average spend | 40 |
| Previous transactions | 1 |
| Account age (days) | 15 |
| Date / time | 2023-01-02 02:15 |
| International | International (1) |

Capture probability ≥ 0.10, risk band Medium/High/Critical, and “Flag for review”. Again, store the real output.

---

## 6. Risk Monitoring

**File:** `06_risk_monitoring.png`

**Where:** sidebar → Risk Monitoring.

**Capture:** KPI row (Low / Medium / High / Critical / Flagged), probability histogram, risk-score histogram, flagged vs not flagged, and the risk-band vs historical-outcome grouped bar.

Caption in the viva: these counts are the **full historical file** scored by the frozen model, not the 1,000-row official test table.

---

## 7. Risk-band distribution

**File:** `07_risk_band_distribution.png`

Crop or zoom the **Risk-band distribution** bar chart from Risk Monitoring so Low / Medium / High / Critical labels are readable.

Optional extra: open `artifacts/model_selection/risk_band_distribution.html` in a browser. That HTML is the **test-set** band chart from Phase 7 (693 / 150 / 126 / 31). If you use it, say “untouched test set” in the caption.

---

## 8. Project folder structure

**File:** `08_project_folder_structure.png`

In File Explorer, open `C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection` and expand:

- `src/`
- `dashboard/`
- `data/raw/`
- `models/production/`
- `reports/`
- `tests/`

Do not expand `.venv`, `.git`, or `requirements_files/` Power BI internals.

---

## 9. Terminal showing pytest results

**File:** `09_pytest_results.png`

```powershell
cd C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection
.\.venv\Scripts\Activate.ps1
pytest -q
```

Expected last line similar to: `110 passed`. Capture the full summary. Do not paste a fake count.

---

## 10. Terminal showing Streamlit launch

**File:** `10_streamlit_launch.png`

```powershell
streamlit run dashboard/app.py
```

Capture the terminal text that includes `You can now view your Streamlit app in your browser` and the Local URL.

---

## Capture tips (Windows)

- Use **Win + Shift + S** (Snipping Tool) or browser full-page screenshot.
- Use a wide window so KPI cards do not wrap into unreadable stacks.
- Dark UI is fine; do not restyle the app for screenshots.
- Never overlay fake metrics, never crop away the “not proof of fraud” disclaimer.

## What this phase does **not** include

No PNG files were generated automatically. Empty `artifacts/screenshots/` is acceptable until you capture the ten shots above.
