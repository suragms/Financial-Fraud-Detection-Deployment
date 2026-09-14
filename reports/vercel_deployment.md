# Vercel deployment — Project Foresight

This document prepares a **public web demo** of the frozen fraud-risk pipeline. It does **not** replace the local Streamlit dashboard.

Local Streamlit (unchanged):

```powershell
streamlit run dashboard/app.py
```

Vercel does **not** run Streamlit.

Production model (frozen, 7,155 bytes): `models/production/final_fraud_pipeline.joblib`  
Threshold: read from `models/production/model_metadata.json` (**0.10**)  
Family: Logistic Regression + Class Weight, sigmoid calibration

---

## 1. Vercel architecture

```
                 VERCEL
                    |
           ┌────────┴────────┐
           |                 |
      Web Frontend       Python API
      (web/)             (api/index.py, Flask)
           |                 |
           └────────┬────────┘
                    |
           Frozen ML Pipeline
                    |
     final_fraud_pipeline.joblib
                    |
     Probability / Risk / Prediction
```

Same-origin by design: the Flask app serves `/`, `/styles.css`, `/app.js`, `/api/health`, and `/api/predict`. The browser never needs a hardcoded `localhost` URL.

The API loads the joblib and metadata **once per function instance** (`lru_cache`). It does not call `.fit(`, does not calibrate, and does not search for a new threshold.

---

## 2. Why Streamlit is not run on Vercel

Streamlit is a long-running interactive server (`streamlit run ...`). Vercel Python hosting is **serverless request/response**. There is no durable Streamlit process, WebSocket session model, or supported `streamlit run` runtime here.

The Streamlit app remains the local/internship dashboard (EDA, model performance, risk monitoring). The Vercel site is a thin prediction demo over the same frozen `predict_transaction()` path.

---

## 3. API architecture

| Item | Detail |
| --- | --- |
| Framework | Flask (`api/index.py`, WSGI `app`) |
| `GET /api/health` | `{ status, model, calibration, threshold }` |
| `POST /api/predict` | Scores one transaction via `src.models.predict.predict_transaction` |
| Threshold | `model_metadata.json` → `production_threshold` |
| Feature engineering | Existing `src/features/feature_engineering.py` |
| Errors | `{ "error": "..." }` only; no stack traces, no filesystem paths |
| Not exposed | Raw CSV, customer IDs, training code, model internals |

Request body uses exactly `FEATURE_INPUT_COLUMNS`:

`Transaction_Date`, `Transaction_Amount`, `Average_Spend`, `Previous_Transactions`, `Account_Age_Days`, `Is_International`, `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`

`Customer_ID`, `Transaction_ID`, `Fraudulent`, and `Suspicious_Keyword` are stripped if present and never sent into X.

`Transaction_Date` format: `dd-mm-YYYY HH:MM` (example `15-06-2023 14:30`).

Invalid amounts, NaN/infinity (as JSON strings), negatives, missing fields, bad dates, and unknown categories return HTTP 400.

---

## 4. Local testing

From the project root, with the virtual environment active:

```powershell
python -m pip install -r requirements.txt
python api/index.py
```

Flask listens on `http://127.0.0.1:8000`. Open that URL for the frontend.

Health:

```powershell
curl http://127.0.0.1:8000/api/health
```

Low-risk example:

```powershell
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d "{\"Transaction_Date\":\"15-06-2023 14:30\",\"Transaction_Amount\":50,\"Average_Spend\":95,\"Previous_Transactions\":20,\"Account_Age_Days\":800,\"Is_International\":0,\"Merchant_Category\":\"Food\",\"Payment_Method\":\"Debit Card\",\"Device_Type\":\"Mobile\",\"Location\":\"Mumbai\"}"
```

Higher-risk example (international, young account):

```powershell
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d "{\"Transaction_Date\":\"02-01-2023 02:15\",\"Transaction_Amount\":250,\"Average_Spend\":40,\"Previous_Transactions\":1,\"Account_Age_Days\":15,\"Is_International\":1,\"Merchant_Category\":\"Travel\",\"Payment_Method\":\"Credit Card\",\"Device_Type\":\"POS\",\"Location\":\"Delhi\"}"
```

Invalid amount:

```powershell
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d "{\"Transaction_Date\":\"15-06-2023 14:30\",\"Transaction_Amount\":-10,\"Average_Spend\":95,\"Previous_Transactions\":20,\"Account_Age_Days\":800,\"Is_International\":0,\"Merchant_Category\":\"Food\",\"Payment_Method\":\"Debit Card\",\"Device_Type\":\"Mobile\",\"Location\":\"Mumbai\"}"
```

Expected invalid body: `{ "error": "Invalid transaction amount" }`.

Automated tests (includes this API and Streamlit smoke):

```powershell
python -m pytest -q
```

---

## 5. GitHub setup

1. Confirm the frozen files are in the repository:
   - `models/production/final_fraud_pipeline.joblib`
   - `models/production/model_metadata.json`
   - `api/index.py`, `web/`, `src/`, `vercel.json`
2. Push the branch to GitHub (do not commit `.venv/` or `.vercel/`).
3. Do **not** commit `data/raw/` into the serverless bundle as a downloadable public dataset if you can avoid it. The Vercel function config excludes `data/raw/**`. Keep the CSV in Git for local Streamlit if that is already how the project is stored.

---

## 6. Vercel project setup

1. Sign in at [vercel.com](https://vercel.com) with GitHub.
2. **Add New Project** → import this repository.
3. Framework: leave auto-detect (Flask / Python).
4. Root directory: repository root (not `dashboard/`).
5. Environment variables: none required for inference.
6. Deploy.

Do not set the start command to `streamlit run`.

---

## 7. Deployment commands

Install the Vercel CLI if you want CLI deploys:

```powershell
npm install -g vercel
```

Preview deployment from the project root (logged-in Vercel account):

```powershell
vercel
```

Production:

```powershell
vercel --prod
```

First run will ask which Vercel account/project to link. Accept the project root. Do not override the output directory to `dashboard/`.

---

## 8. Environment configuration

None required. Threshold and model family come from `model_metadata.json`.

Optional later (not enabled): `FORESIGHT_CORS_ORIGINS` is **not** used. CORS is omitted because the frontend and API are same-origin. Do not add `Access-Control-Allow-Origin: *` unless a separate frontend host is introduced on purpose.

---

## 9. Troubleshooting

| Symptom | What to check |
| --- | --- |
| 503 `Production model is unavailable` | `models/production/final_fraud_pipeline.joblib` missing from the GitHub repo or excluded from the function bundle |
| 400 `Invalid transaction date` | Send `dd-mm-YYYY HH:MM`, not ISO, from raw API clients. The HTML form converts `datetime-local` automatically |
| 400 `Invalid merchant category` | Use the closed lists from the training data (see the form dropdowns) |
| Frontend cannot call the API | Open the site origin that serves Flask (`/` and `/api/predict`). Do not open `web/index.html` as a `file://` page. On some Windows machines port 5000 is reserved; the local server uses **8000** |
| Deploy size / install timeout | The joblib is ~7 KB. Heavy local extras (`requirements_files/`, Streamlit) are excluded from the function via `vercel.json`. Root `requirements.txt` still lists Streamlit for local use; if a Vercel install hits a size or time limit, keep the frozen model and install only Flask, pandas, numpy, scikit-learn, and joblib for the function |
| Streamlit “broken” after this work | It should not be. Re-run `streamlit run dashboard/app.py` and `python -m pytest tests/test_dashboard.py tests/test_phase9.py -q` |

---

## 10. Production limitations

- Internship / portfolio demo, not a bank-grade fraud engine.
- Precision on the frozen test set is **23.78%**; flags need human review.
- Serverless cold start loads sklearn + the 7 KB pipeline; keep requests small (one transaction).
- Unknown categories are **rejected** by the public API (clean error). The underlying sklearn encoder could ignore them; the API is stricter on purpose.
- The Vercel app does not ship EDA, risk monitoring tables, or the raw CSV download.
- Hobby-plan execution time is limited; `maxDuration` is 30 seconds, which is enough for single-row inference.
- Model artifact size at freeze: **7,155 bytes (0.007 MB)** — well under Vercel function size limits. Do not compress or replace it.

---

## Model file size check

| File | Size |
| --- | --- |
| `models/production/final_fraud_pipeline.joblib` | 7,155 bytes |
| `models/production/model_metadata.json` | 4,792 bytes |

No resize, recompression, or retraining was performed.
