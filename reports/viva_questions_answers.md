# Viva questions and answers — Project Foresight

Use simple spoken English. Numbers below are the **verified project figures**. Do not invent better metrics in the viva.

Official production operating point (untouched test set, threshold **0.10**):

Accuracy 0.7430 · Precision 0.2378 · Recall 0.7604 · F1 0.3623 · ROC-AUC 0.7646 · PR-AUC 0.2620  
TP 73 · FP 234 · TN 670 · FN 23 · 1,000 test rows · 96 test frauds

Phase 6 comparison used the **default 0.50** cut **before** calibration. Those numbers are historical, not the final dashboard metrics.

---

## A. Basic questions

### 1. What is financial fraud detection?

It is the task of spotting payment or account activity that is likely dishonest — stolen cards, unauthorized transfers, and similar abuse — so a human or system can review it. In this project we estimate **fraud risk**, we do not declare legal proof of fraud.

### 2. Why did you choose this project?

Fraud is a real business problem, the data is **highly imbalanced**, and accuracy is a trap. It was a good internship topic to practice leakage-safe machine learning, calibration, and a dashboard.

### 3. What dataset did you use?

`data/raw/financial_fraud_detection_dataset.csv` — 14 columns, target `Fraudulent`. Other files under `requirements_files/` were not used as the application dataset.

### 4. How many transactions?

**5,000**.

### 5. How many fraud cases?

**482** fraud and **4,518** legitimate. Fraud rate **9.64%**.

### 32. What is Streamlit?

A Python library for small web apps. We used it for the dashboard. Command: `streamlit run dashboard/app.py`.

### 34. Does the dashboard retrain the model?

**No.** It loads `models/production/final_fraud_pipeline.joblib` and only runs inference.

---

## B. Dataset questions

### 6. What is class imbalance?

One class is much rarer than the other. Here about **1 in 10** rows is fraud. A dummy “always legitimate” model is already about **90%** accurate.

### Train/test split?

`test_size=0.20`, `stratify=y`, `random_state=42`. Train **4,000** rows (**386** fraud). Test **1,000** rows (**96** fraud). No overlapping `Transaction_ID`.

### Date range?

2023-01-01 to 2024-02-21.

### Missing values / duplicate IDs?

**0** missing cells. **0** duplicate `Transaction_ID`.

### Raw dataset hash?

MD5 `9a4a90ce2e07a717b4289dc95a71663c`.

---

## C. EDA questions

### What did EDA show about international transactions?

Historical fraud rate **36.96%** international (163/441) vs **7.00%** domestic (319/4559). That is an association, not a causal claim.

### Is amount a strong fraud signal?

No. Correlation with the target is about **0.0116**. Mean amounts are close (fraud 81.99 vs legitimate 78.88).

### Time of day?

Overnight hours 00:00–05:00 show higher observed fraud rates than later hours.

### Keyword column in EDA?

`Suspicious_Keyword=Yes` has **47.31%** historical fraud vs **7.57%** when No. It was **excluded from the model** because provenance/timing is undocumented.

### Customers?

**3,847** unique `Customer_ID`; many customers have only one row, so customer-level history was not used as a feature.

---

## D. Preprocessing questions

### What preprocessing did you use?

**RobustScaler** on numeric columns and **OneHotEncoder** (`handle_unknown="ignore"`, dense output) on Merchant, Payment, Device, Location.

### When was it fitted?

**Training fold only.** The test fold never fits scalers or encoders.

### Did you drop outliers?

**No.** Extreme amounts can be fraud. Rows were not deleted for modeling.

### Date format?

`%d-%m-%Y %H:%M`. Invalid dates raise an error.

---

## E. Feature engineering questions

### 30. What is feature engineering?

Creating extra columns from existing fields so the model can use them. We keep it **row-local**: one transaction does not look at other customers’ labels.

### How many model features?

**18**.

### 31. What is cyclical encoding?

Hour 23 and hour 0 are close in time. We store `sin` and `cos` of hour/24 and weekday/7 so the model sees that circle, not a fake jump from 23 to 0.

### `amount_to_average_ratio`?

`Transaction_Amount / Average_Spend`. If average spend is 0, the ratio is **0**.

### 27. Why was Customer_ID excluded?

It is an identifier. Using it can leak identity and does not generalize to new customers. Many IDs appear once.

### 28. Why was Transaction_ID excluded?

It is a unique key. It has no repeatable pattern for future transactions.

### 29. Why was Suspicious_Keyword excluded?

EDA shows a strong association, but we do not know if the keyword was filled **after** fraud was known. Using it could leak the label. The dashboard does not collect it.

### Why was raw Transaction_Date excluded from X?

The date string is parsed into hour, weekday, month, year, and cyclical terms. The raw timestamp is not a model column.

---

## F. Machine learning questions

### Which algorithms did you try?

Logistic Regression, Decision Tree, Random Forest, XGBoost. Each with **SMOTE** and with **class weight**. Eight candidates.

### 13. Why did you choose Logistic Regression?

On the Phase 6 test table it had the **highest F1 (0.3668)** and **tied-highest recall (0.7604)** with Random Forest class weight, with **better precision (0.2417 vs 0.2325)** and a simpler, more explainable model.

### 14. What is class weighting?

The loss gives more weight to the rare fraud class (`class_weight="balanced"`). For XGBoost we used `scale_pos_weight` from the **train** fold only.

### 15. What is SMOTE?

Synthetic Minority Over-sampling. It creates extra minority samples in the **transformed training** space. It was never applied to the test set.

### 16. Why was SMOTE not selected for the final model?

SMOTE Logistic Regression had slightly higher PR-AUC (0.2681 vs 0.2620) but **lower F1 and recall**. SMOTE XGBoost looked accurate (**0.8820**) but recall was only **0.0938**. Class-weight logistic regression was the better fraud-review candidate.

### Random state?

**42** everywhere we control seeds.

### 17. What is probability calibration?

Making predicted probabilities closer to true frequencies. Uncalibrated class-weighted logistic regression had inner-val mean predicted **0.339** vs observed **0.096**. Sigmoid `CalibratedClassifierCV` (5-fold, `ensemble=False`) cut Brier from **0.1567** to **0.0706**. Test labels were not used to fit calibration.

---

## G. Model evaluation questions

### 7. Why is accuracy misleading?

Legitimate rows dominate. Dummy always-legitimate on the **test set** is **90.40%** accurate with **0%** recall. Production accuracy **74.30%** is lower but recall is **76.04%**.

### 8. What is precision?

Of transactions the model flags, what share are actually fraud. Final test precision **23.78%** — about one in four flags is fraud.

### 9. What is recall?

Of actual frauds, what share did we catch. Final test recall **76.04%** — **73 of 96**.

### 10. What is F1-score?

Harmonic mean of precision and recall. Final test F1 **0.3623**.

### 11. What is ROC-AUC?

Area under the ROC curve; ranking of fraud vs legitimate across thresholds. Final **0.7646**.

### 12. What is PR-AUC?

Area under precision–recall. Better than ROC when the positive class is rare. Final **0.2620**. Prevalence is 9.60% on the test fold, so a random ranking would sit near that baseline.

### Did you use the test set to choose the model family?

Phase 6 compared eight models **once** on the held-out test fold. Threshold and calibration used **training** data only. The official production metrics are a **second** one-shot score after freeze (threshold 0.10). We do not keep retuning on test.

### Dummy model?

Always Legitimate: Accuracy 0.9040, Precision 0, Recall 0, F1 0, ROC-AUC 0.5000, PR-AUC 0.0960.

---

## H. Fraud detection questions

### 18. Why is the threshold 0.10?

After sigmoid calibration, probabilities sit near the ~10% fraud rate. On **training out-of-fold** scores, **0.10** was the only grid point with recall ≥ 0.70 and FPR ≤ 0.35 (train OOF recall 0.8161, FPR 0.2438).

### 19. Why not threshold 0.50?

0.50 is a default, not a business rule. After calibration most true frauds fall below 0.50, so recall collapses. 0.15 had a slightly higher train F1 but recall **0.6658**, below the 0.70 screen.

### 20. What is a false positive?

A **legitimate** transaction the model flags. Test **FP = 234**.

### 21. What is a false negative?

A **fraud** the model calls legitimate. Test **FN = 23**.

### 22. Which is more important in fraud detection?

Missing fraud (false negative) is usually more costly per case. False positives still matter because they create review work. We targeted high recall with a cap on FPR, not zero FPs.

### 35. How is a transaction classified?

Features → frozen pipeline `predict_proba` → fraud probability. If probability **≥ 0.10**, `predicted_label = 1` (flag for review), else 0.

### 36. What happens when probability >= 0.10?

It is **flagged for review**. It is not automatically blocked and it is not proven fraud.

### 23. What is risk score?

`risk_score = predicted_probability × 100`, clipped to 0–100. A model **risk indicator**.

### 24. What are the risk bands?

| Score | Band |
| --- | --- |
| 0–9 | Low Risk |
| 10–19 | Medium Risk |
| 20–39 | High Risk |
| 40–100 | Critical Risk |

Low is below the 0.10 threshold. Bands were aligned to calibration, **not** fitted on test labels.

---

## I. Dashboard questions

### 33. How does the dashboard work?

Five pages in `dashboard/app.py`. It caches the frozen pipeline, the raw CSV (hash-checked), official test metrics, and scored historical rows. Prediction uses `predict_transaction()`.

### Pages?

1. Overview — dataset KPIs and simple EDA  
2. Fraud Analytics — filterable historical rates  
3. Model Performance — frozen Phase 7 metrics  
4. Transaction Prediction — score one new row  
5. Risk Monitoring — bands, flags, downloadable table  

### Can the user change the threshold in the UI?

**No.** Threshold is frozen at **0.10** in metadata.

### Does prediction send Customer_ID or the keyword?

**No.** Those fields are not on the form.

---

## J. Technical questions

### 25. What is data leakage?

When information that would not be available at prediction time — especially the label or future knowledge — sneaks into training or features. Metrics then look too good.

### 26. How did you prevent leakage?

Split first. Fit preprocess on train only. No target encoding. No full-file customer aggregates. Exclude IDs, keyword, and `Fraudulent` from X. Calibration and threshold on train/OOF only. Dashboard never calls `.fit(`.

### Production artifact?

`models/production/final_fraud_pipeline.joblib`  
MD5 `5d08c124ddbb452f7a93f6982d02240a`

### How do unknown merchants get handled?

One-hot `handle_unknown="ignore"`: extra categories become zeros; the app should not crash.

### Invalid inputs?

Negative amount, NaN, inf, empty frame, or a bad date raise a validation error. The dashboard shows a short message, not a stack trace.

### 39. How did you test the project?

`python run_validation.py` for the CSV schema. `pytest -q` for unit/integration tests (dataset, EDA, features, training, evaluation, Phase 7, dashboard AppTest, Phase 9 inference). **110 passed**, 0 failed, 0 skipped at Phase 9 freeze.

### 40. How do you know your results are reproducible?

Fixed seed 42, frozen CSV and joblib hashes, one-shot test evaluation written to CSV, pytest guards those hashes. Re-running training is **not** required for the dashboard.

---

## K. Project limitations

### 37. What are the limitations?

- Precision **23.78%**: most flags are legitimate.  
- **23** of **96** test frauds missed.  
- Only **96** test positives: metrics have sampling noise.  
- Short date window; we do not claim seasonality.  
- No live bank feed, no Kafka/Spark.  
- Unknown categories are ignored, not rejected.  
- **Not** 100% accurate and **not** a certified production banking system.

### Is the model production-ready for a bank?

**No.** Metadata even stores `production_ready_claim: false`. It is an internship / prototype pipeline.

---

## L. Future scope

### 38. What would you improve in the future?

Possible later work, **not implemented now**: training-only customer history, documented keyword timing, cost-sensitive thresholds still chosen on train/OOF, analyst case workflow. We would **not** retune on the current test fold.

---

## Extra questions examiners often ask

### Why not neural networks / deep learning?

5,000 rows is small. Logistic regression was competitive, simpler, and easier to explain in a viva.

### Why RobustScaler not StandardScaler?

Transaction amounts can have heavy tails. RobustScaler uses median and IQR, so a few huge amounts do not dominate scaling.

### Why F1 over accuracy for selection?

F1 balances precision and recall on the fraud class. Accuracy can be high while missing fraud (see XGBoost SMOTE).

### Why is PR-AUC more relevant than ROC-AUC here?

ROC can look acceptable when negatives dominate. PR focuses on the fraud class. Our PR-AUC **0.2620** is above the 9.6% prevalence baseline but still modest.

### Did calibration change ranking?

Sigmoid is monotonic, so ROC-AUC and PR-AUC stayed **0.7646 / 0.2620**. Class metrics shifted slightly (FP 229 → 234 at the new threshold).

### Can I download predictions?

Risk Monitoring has **Download scored results CSV**. It will not overwrite `data/raw/financial_fraud_detection_dataset.csv`.

### What libraries?

pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, joblib, Plotly, Streamlit, pytest.

### What is the project title?

Financial Fraud Detection & Risk Analytics System Using Machine Learning with an Interactive Streamlit Dashboard.

---

## Quick number card (keep this in front of you)

| Item | Value |
| --- | --- |
| Rows / frauds | 5,000 / 482 |
| Fraud rate | 9.64% |
| Features | 18 |
| Final model | Logistic Regression + Class Weight |
| Calibration | Sigmoid, 5-fold, ensemble=False |
| Threshold | 0.10 |
| Test recall / precision | 76.04% / 23.78% |
| Test TP / FP / FN / TN | 73 / 234 / 23 / 670 |
| Raw MD5 | 9a4a90ce2e07a717b4289dc95a71663c |
| Model MD5 | 5d08c124ddbb452f7a93f6982d02240a |
