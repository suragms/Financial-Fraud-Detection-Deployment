# Phase 7 Model Selection, Calibration, and Risk Scoring

## 1. Objective

Freeze a production inference pipeline from the Phase 6 selected candidate.
Calibration and the operating threshold are chosen from training data only.
The untouched 1,000-row Phase 6 test fold is scored once after those decisions are frozen.
The risk score is a model risk indicator, not proof that a transaction is fraudulent.

## 2. Phase 6 selected candidate

- Model: Logistic Regression
- Strategy: Class Weight
- Artifact: `models/baseline/logistic_regression_class_weight.joblib`
- Phase 6 default-threshold metrics: Accuracy=0.7480, Precision=0.2417, Recall=0.7604, F1=0.3668, ROC-AUC=0.7646, PR-AUC=0.2620
- Phase 6 confusion: TN=675, FP=229, FN=23, TP=73

## 3. Calibration methodology

- Recreate the Phase 5/6 split (`test_size=0.20`, `stratify=y`, `random_state=42`).
- Hold out 20% of the **training** fold only (3200 inner-train / 800 inner-val, 77 inner-val frauds).
- Fit an uncalibrated clone of Logistic Regression + Class Weight on inner-train.
- Fit `CalibratedClassifierCV` with `method='sigmoid'` and `method='isotonic'`, `cv=5`, `ensemble=False`, on inner-train.
- Compare Brier score and log loss on inner-val. The test fold is not used.
- Sigmoid is the primary option. Isotonic is diagnostic only unless the data clearly support it.

## 4. Calibration result

- Inner-val uncalibrated: Brier=0.156707, log_loss=0.490549, mean predicted=0.3391, observed rate=0.0963
- Inner-val sigmoid: Brier=0.070583, log_loss=0.242728, mean predicted=0.0974
- Inner-val isotonic: Brier=0.070003, log_loss=0.239839, mean predicted=0.0966
- Selected calibration method: **sigmoid**

Sigmoid calibration reduced inner-validation Brier from 0.156707 to 0.070583 (drop 0.086124) and log loss from 0.490549 to 0.242728. The improvement is large enough to justify wrapping the selected Logistic Regression + Class Weight pipeline with CalibratedClassifierCV (method='sigmoid', cv=5, ensemble=False) fit on the full training fold.

Isotonic was evaluated as a diagnostic only. The inner validation fold has 800 rows and 77 frauds, which is thin for a non-parametric calibrator, so isotonic was not selected.

## 5. Threshold-selection methodology

- Out-of-fold `predict_proba` from 5-fold stratified CV on the 4,000-row training fold.
- The same architecture as the frozen production estimator (calibrated or not).
- Grid: 0.10 to 0.90 in steps of 0.05.
- Primary screen: recall at least 0.70 and FPR at most 0.35, then maximize F1.
- Accuracy is not used to pick the threshold.
- Test labels are not inspected during this search.

## 6. Threshold comparison

Training out-of-fold metrics:

| threshold | Accuracy | Precision | Recall | F1 | False_Positive_Rate | False_Negative_Rate | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.7620 | 0.2634 | 0.8161 | 0.3982 | 0.2438 | 0.1839 | 2733 | 881 | 71 | 315 |
| 0.1500 | 0.8113 | 0.2911 | 0.6658 | 0.4050 | 0.1732 | 0.3342 | 2988 | 626 | 129 | 257 |
| 0.2000 | 0.8450 | 0.2990 | 0.4508 | 0.3595 | 0.1129 | 0.5492 | 3206 | 408 | 212 | 174 |
| 0.2500 | 0.8752 | 0.3435 | 0.3212 | 0.3320 | 0.0656 | 0.6788 | 3377 | 237 | 262 | 124 |
| 0.3000 | 0.8935 | 0.4048 | 0.2202 | 0.2852 | 0.0346 | 0.7798 | 3489 | 125 | 301 | 85 |
| 0.3500 | 0.9008 | 0.4650 | 0.1891 | 0.2689 | 0.0232 | 0.8109 | 3530 | 84 | 313 | 73 |
| 0.4000 | 0.9042 | 0.5120 | 0.1658 | 0.2505 | 0.0169 | 0.8342 | 3553 | 61 | 322 | 64 |
| 0.4500 | 0.9040 | 0.5096 | 0.1373 | 0.2163 | 0.0141 | 0.8627 | 3563 | 51 | 333 | 53 |
| 0.5000 | 0.9050 | 0.5326 | 0.1269 | 0.2050 | 0.0119 | 0.8731 | 3571 | 43 | 337 | 49 |
| 0.5500 | 0.9048 | 0.5333 | 0.1036 | 0.1735 | 0.0097 | 0.8964 | 3579 | 35 | 346 | 40 |
| 0.6000 | 0.9048 | 0.5385 | 0.0907 | 0.1552 | 0.0083 | 0.9093 | 3584 | 30 | 351 | 35 |
| 0.6500 | 0.9045 | 0.5400 | 0.0699 | 0.1239 | 0.0064 | 0.9301 | 3591 | 23 | 359 | 27 |
| 0.7000 | 0.9045 | 0.5625 | 0.0466 | 0.0861 | 0.0039 | 0.9534 | 3600 | 14 | 368 | 18 |
| 0.7500 | 0.9050 | 0.6667 | 0.0311 | 0.0594 | 0.0017 | 0.9689 | 3608 | 6 | 374 | 12 |
| 0.8000 | 0.9040 | 0.7500 | 0.0078 | 0.0154 | 0.0003 | 0.9922 | 3613 | 1 | 383 | 3 |
| 0.8500 | 0.9032 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 1.0000 | 3613 | 1 | 386 | 0 |
| 0.9000 | 0.9035 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 3614 | 0 | 386 | 0 |

## 7. Selected production threshold

- **0.10**
- Training OOF: Precision=0.2634, Recall=0.8161, F1=0.3982, FPR=0.2438, FNR=0.1839
- Training OOF confusion: TN=2733, FP=881, FN=71, TP=315
- Rule: recall at least 0.70 and FPR at most 0.35, then maximize F1 / recall / precision and minimize FPR

Threshold 0.10 was selected from training out-of-fold probabilities (5-fold CV on 4000 training rows, 386 training frauds). It has training OOF recall 0.8161 and FPR 0.2438, with F1 0.3982. The default 0.50 cut was not adopted automatically. Selection rule: recall at least 0.70 and FPR at most 0.35, then maximize F1 / recall / precision and minimize FPR. On the same training OOF table, 0.15 had a slightly higher F1 but recall below 0.70, so it failed the primary fraud-recall screen.

## 8. Risk-score methodology

`risk_score = predicted fraud probability x 100`, clipped to 0-100.
This is a model risk indicator for review prioritization. It is not a determination of fraud.

## 9. Risk bands

| Band | Score range | Meaning |
| --- | --- | --- |
| Low Risk | 0-9 | Below the production review threshold |
| Medium Risk | 10-19 | Flagged review range |
| High Risk | 20-39 | Higher calibrated fraud probability |
| Critical Risk | 40-100 | Highest calibrated fraud probability |

Sigmoid calibration mapped probabilities onto the training prevalence scale (inner-val mean predicted ~0.10). The original 0-29 Low band would have placed almost every flagged review in Low Risk. Bands were therefore aligned to the frozen training threshold with round cut-points: Low is below-threshold, Medium/High/Critical partition the flagged range. These cuts were not estimated from test labels.

## 10. Final model architecture

- Classifier: Logistic Regression (`class_weight='balanced'`)
- Strategy: Class Weight
- Features (18): Transaction_Amount, Average_Spend, Previous_Transactions, Account_Age_Days, amount_to_average_ratio, transaction_hour, transaction_day_of_week, transaction_month, transaction_year, hour_sin, hour_cos, dow_sin, dow_cos, Is_International, Merchant_Category, Payment_Method, Device_Type, Location
- Numeric preprocess: RobustScaler on Transaction_Amount, Average_Spend, Previous_Transactions, Account_Age_Days, amount_to_average_ratio, transaction_hour, transaction_day_of_week, transaction_month, transaction_year, hour_sin, hour_cos, dow_sin, dow_cos, Is_International
- Categorical preprocess: OneHotEncoder(handle_unknown='ignore', sparse_output=False) on Merchant_Category, Payment_Method, Device_Type, Location
- Calibration: sigmoid
- Production threshold: 0.10
- Excluded from X: Transaction_ID, Customer_ID, Suspicious_Keyword, Fraudulent, Transaction_Date
- Artifact: `models/production/final_fraud_pipeline.joblib`

## 11. Final untouched-test evaluation

- Test rows: 1000; fraud=96; legitimate=904; rate=9.60%
- Threshold: 0.10
- Accuracy=0.7430
- Precision=0.2378
- Recall=0.7604
- F1=0.3623
- ROC-AUC=0.7646
- PR-AUC=0.2620
- FPR=0.2588
- FNR=0.2396
- TN=670, FP=234, FN=23, TP=73
- Flagged as fraud: 307 (30.70%)
- Classified legitimate: 693 (69.30%)

These test numbers were not used to change the threshold, calibration, or model.

### Test-set risk score snapshot

- Minimum: 1.0541
- Maximum: 86.2122
- Mean: 9.7542
- Median: 4.0275

| risk_band | score_range | count | percentage | mean_risk_score | fraud_count | fraud_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Low Risk | 0-9 | 693 | 0.6930 | 3.5960 | 23 | 0.0332 |
| Medium Risk | 10-19 | 150 | 0.1500 | 14.9054 | 35 | 0.2333 |
| High Risk | 20-39 | 126 | 0.1260 | 25.7877 | 26 | 0.2063 |
| Critical Risk | 40-100 | 31 | 0.0310 | 57.3257 | 12 | 0.3871 |

Example rows use Transaction_ID only. Customer identifiers are not shown.

| Transaction_ID | risk_score | risk_band | predicted_label |
| --- | --- | --- | --- |
| T101854 | 12.9549 | Medium Risk | 1 |
| T104390 | 6.0158 | Low Risk | 0 |
| T103651 | 2.9225 | Low Risk | 0 |
| T102255 | 2.3179 | Low Risk | 0 |
| T100894 | 3.4339 | Low Risk | 0 |
| T101286 | 3.2775 | Low Risk | 0 |
| T100486 | 6.9687 | Low Risk | 0 |
| T100343 | 17.8066 | Medium Risk | 1 |
| T101174 | 6.8877 | Low Risk | 0 |
| T100065 | 3.1263 | Low Risk | 0 |
| T104486 | 14.7445 | Medium Risk | 1 |
| T104188 | 2.6275 | Low Risk | 0 |

## 12. Limitations

- 96 test frauds: interval estimates around recall/precision are wide.
- Class-weighted logistic regression probabilities are not a causal fraud proof.
- False positives create review workload; false negatives are missed fraud.
- No dashboard, SQLite, or deployment is included in this phase.

## 13. Reproducibility information

- random_state: 42
- dataset: `financial_fraud_detection_dataset.csv`
- raw MD5: `9a4a90ce2e07a717b4289dc95a71663c`
- production model MD5: `5d08c124ddbb452f7a93f6982d02240a`
- threshold CSV MD5: `3db01d1360d5e1d8ec084ec0a72cd7dd`
- final metrics CSV MD5: `558624d04767ca2ab9a6af3633467e47`
- risk-band CSV MD5: `5bbf5ebc5c002d96d7fa9e062816033f`

## 14. Data leakage safeguards

- Split first; test never resampled.
- Preprocess remains the Phase 5 train-fit transformer inside the saved pipeline.
- Calibration compared and, if used, fitted on training rows only.
- Threshold chosen from training out-of-fold probabilities only.
- Customer_ID, Transaction_ID, Suspicious_Keyword, Fraudulent, and raw Transaction_Date are not model features.

## 15. Conclusion

The production wrapper is Logistic Regression + Class Weight with calibration=sigmoid and threshold=0.10. On the untouched test set it scores Recall=0.7604, Precision=0.2378, F1=0.3623, ROC-AUC=0.7646, PR-AUC=0.2620, TP=73, FP=234, TN=670, FN=23. These test metrics were not used for further tuning.
