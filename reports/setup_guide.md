# Windows setup guide — Project Foresight

Use **Windows PowerShell**. Commands assume the project lives at:

`C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection`

If your path is different, `cd` there first. Do **not** retrain, retune the threshold, or replace files under `data/raw/` or `models/production/`.

---

## 1. Python environment

You need **Python 3.10+** with `pip`.

```powershell
python --version
```

If `python` is not found:

```powershell
py --version
py -3 --version
```

Use the same launcher (`python` or `py -3`) in every step below.

Install Python from https://www.python.org/downloads/ if needed. During setup, enable **Add python.exe to PATH**.

---

## 2. Create a virtual environment

```powershell
cd C:\Users\SURAG\Documents\zidio\Financial_Fraud_Detection
python -m venv .venv
```

This creates `.venv\` in the project folder. Do not commit it.

---

## 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)`.

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Alternative (Command Prompt, not PowerShell):

```cmd
.venv\Scripts\activate.bat
```

---

## 4. Install requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Packages include pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Streamlit, Plotly, joblib, pytest.

---

## 5. Validate the dataset

```powershell
python run_validation.py
```

Expected:

- file `data/raw/financial_fraud_detection_dataset.csv`
- 5,000 rows, 14 columns
- 482 fraud, 4,518 legitimate
- validation **passed**

Raw MD5 must remain:

`9a4a90ce2e07a717b4289dc95a71663c`

Do not edit that CSV. Do not point the app at `requirements_files/`.

---

## 6. Run tests

```powershell
pytest -q
```

Expected at the Phase 9/10 freeze: **110 passed**, 0 failed, 0 skipped.

Optional verbose:

```powershell
pytest -q tests
```

---

## 7. Run the dashboard

```powershell
streamlit run dashboard/app.py
```

Open the Local URL (usually `http://localhost:8501`).

The app loads `models/production/final_fraud_pipeline.joblib` (MD5 `5d08c124ddbb452f7a93f6982d02240a`). It does **not** call training.

Stop the server with **Ctrl+C** in the same terminal.

---

## Optional: project hash check

From the project root, after activation:

```powershell
python -c "import hashlib, pathlib; p=pathlib.Path('data/raw/financial_fraud_detection_dataset.csv'); print(hashlib.md5(p.read_bytes()).hexdigest())"
python -c "import hashlib, pathlib; p=pathlib.Path('models/production/final_fraud_pipeline.joblib'); print(hashlib.md5(p.read_bytes()).hexdigest())"
```

Expected:

```
9a4a90ce2e07a717b4289dc95a71663c
5d08c124ddbb452f7a93f6982d02240a
```

---

## Common errors and simple fixes

### `Activate.ps1` cannot be loaded because running scripts is disabled

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### `python` is not recognized

Install Python 3 and tick PATH, or use `py -3` instead of `python`.

### `No module named streamlit` / `sklearn` / `xgboost`

The venv is not active, or install did not finish:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### `Production pipeline was not found`

`models/production/final_fraud_pipeline.joblib` is missing. Restore it from version control. **Do not** treat this as a reason to retrain unless a supervisor explicitly asks. Phase 10 documentation assumes the frozen file is present.

### Dashboard: “The raw dataset hash has changed”

`data/raw/financial_fraud_detection_dataset.csv` was edited or replaced. Restore the original file. Do not “fix” the hash in code.

### `Address already in use` / port 8501 busy

Streamlit will offer another port, or:

```powershell
streamlit run dashboard/app.py --server.port 8502
```

### `pytest` is not recognized

The virtual environment is not active, or Scripts is not on PATH. Activate `.venv`, then:

```powershell
python -m pytest -q
```

### Pytest collection errors / `Null bytes`

A source file was saved as UTF-16. Re-save as UTF-8. Do not change model artifacts to “fix” this.

### `imblearn` / SMOTE import error

Reinstall from `requirements.txt`. SMOTE is only used in **historical training pipelines**, not by the dashboard.

### Excel or Power BI files under `requirements_files/`

Ignore them for running the app. The dashboard never loads `.pbix` or extra CSVs from that folder.

### Antivirus locks `joblib` or `xgboost.dll`

Allow the project folder, or pause real-time scan briefly while installing.

---

## What you should not do

- Do not run `python run_training.py` or `python run_phase7.py` to “improve” results for the viva.
- Do not change threshold 0.10.
- Do not modify `data/raw/financial_fraud_detection_dataset.csv`.
- Do not claim the model is production banking-grade.
