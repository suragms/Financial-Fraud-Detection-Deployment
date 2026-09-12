# Phase 6 Model Comparison

All figures below were computed by scoring saved Phase 5 pipelines on one untouched test set.
No hyperparameters were changed. The default decision threshold was used. This is a **selected candidate**, not a production model.

## 1. Evaluation methodology

- Load `data/raw/financial_fraud_detection_dataset.csv`
- `build_features()` from Phase 4 (no target, IDs, or Suspicious_Keyword in X)
- Recreate the Phase 5 split: `test_size=0.20`, `stratify=y`, `random_state=42`
- Load eight `joblib` pipelines; do not refit
- `y_pred = predict(X_test)`; `y_proba = predict_proba(X_test)[:, 1]`
- ROC-AUC and PR-AUC use probabilities, not hard labels
- Precision / recall / F1 use `average='binary'` and `zero_division=0`

## 2. Test set

- Rows: 1000
- Fraud count: 96
- Legitimate count: 904
- Fraud rate: 9.60%
- Train rows (not used for scoring): 4000

A classifier that predicts every row as legitimate would already be about 90.40% accurate while catching **zero** fraud. Accuracy alone is therefore not the selection metric.

## 3. Model comparison table

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | SMOTE | 0.7470 | 0.2357 | 0.7292 | 0.3562 | 0.7682 | 0.2681 | 677 | 227 | 26 | 70 |
| Decision Tree | SMOTE | 0.7650 | 0.2186 | 0.5625 | 0.3149 | 0.7576 | 0.2335 | 711 | 193 | 42 | 54 |
| Random Forest | SMOTE | 0.7880 | 0.2434 | 0.5729 | 0.3416 | 0.7588 | 0.2235 | 733 | 171 | 41 | 55 |
| XGBoost | SMOTE | 0.8820 | 0.2250 | 0.0938 | 0.1324 | 0.7584 | 0.2336 | 873 | 31 | 87 | 9 |
| Logistic Regression | Class Weight | 0.7480 | 0.2417 | 0.7604 | 0.3668 | 0.7646 | 0.2620 | 675 | 229 | 23 | 73 |
| Decision Tree | Class Weight | 0.7530 | 0.2351 | 0.6979 | 0.3517 | 0.7639 | 0.2559 | 686 | 218 | 29 | 67 |
| Random Forest | Class Weight | 0.7360 | 0.2325 | 0.7604 | 0.3561 | 0.7577 | 0.2412 | 663 | 241 | 23 | 73 |
| XGBoost | Class Weight | 0.7830 | 0.2531 | 0.6458 | 0.3636 | 0.7509 | 0.2262 | 721 | 183 | 34 | 62 |

### Dummy baseline (not a trained model)

Always Legitimate predicts every test row as class 0. It is included only to show why accuracy is a poor fraud metric.

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always Legitimate | Dummy | 0.9040 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0960 | 904 | 0 | 96 | 0 |

Constant-zero scores yield ROC-AUC=0.5000 (no ranking) and PR-AUC=0.0960 (equal to prevalence when scores are constant). These AUC values are diagnostic, not competitive.

## 4. Confusion matrix summary

Counts are from the 1,000-row test set (TN+FP+FN+TP = 1,000 for every row).

| Model | Strategy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | SMOTE | 677 | 227 | 26 | 70 |
| Decision Tree | SMOTE | 711 | 193 | 42 | 54 |
| Random Forest | SMOTE | 733 | 171 | 41 | 55 |
| XGBoost | SMOTE | 873 | 31 | 87 | 9 |
| Logistic Regression | Class Weight | 675 | 229 | 23 | 73 |
| Decision Tree | Class Weight | 686 | 218 | 29 | 67 |
| Random Forest | Class Weight | 663 | 241 | 23 | 73 |
| XGBoost | Class Weight | 721 | 183 | 34 | 62 |

## 5. ROC-AUC comparison

| Model | Strategy | ROC_AUC |
| --- | --- | --- |
| Logistic Regression | SMOTE | 0.7682 |
| Logistic Regression | Class Weight | 0.7646 |
| Decision Tree | Class Weight | 0.7639 |
| Random Forest | SMOTE | 0.7588 |
| XGBoost | SMOTE | 0.7584 |
| Random Forest | Class Weight | 0.7577 |
| Decision Tree | SMOTE | 0.7576 |
| XGBoost | Class Weight | 0.7509 |

## 6. PR-AUC comparison

PR-AUC is the more informative ranking metric at 9.6% prevalence.

| Model | Strategy | PR_AUC |
| --- | --- | --- |
| Logistic Regression | SMOTE | 0.2681 |
| Logistic Regression | Class Weight | 0.2620 |
| Decision Tree | Class Weight | 0.2559 |
| Random Forest | Class Weight | 0.2412 |
| XGBoost | SMOTE | 0.2336 |
| Decision Tree | SMOTE | 0.2335 |
| XGBoost | Class Weight | 0.2262 |
| Random Forest | SMOTE | 0.2235 |

## 7. Precision / recall tradeoff

| Model | Strategy | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Class Weight | 0.2417 | 0.7604 | 0.3668 | 229 | 23 |
| XGBoost | Class Weight | 0.2531 | 0.6458 | 0.3636 | 183 | 34 |
| Logistic Regression | SMOTE | 0.2357 | 0.7292 | 0.3562 | 227 | 26 |
| Random Forest | Class Weight | 0.2325 | 0.7604 | 0.3561 | 241 | 23 |
| Decision Tree | Class Weight | 0.2351 | 0.6979 | 0.3517 | 218 | 29 |
| Random Forest | SMOTE | 0.2434 | 0.5729 | 0.3416 | 171 | 41 |
| Decision Tree | SMOTE | 0.2186 | 0.5625 | 0.3149 | 193 | 42 |
| XGBoost | SMOTE | 0.2250 | 0.0938 | 0.1324 | 31 | 87 |

## 8. False-positive analysis

A false positive is a legitimate test transaction flagged as fraud. The test set has 904 legitimate rows.

- XGBoost (SMOTE): FP=31 (precision=0.2250)
- Random Forest (SMOTE): FP=171 (precision=0.2434)
- XGBoost (Class Weight): FP=183 (precision=0.2531)
- Decision Tree (SMOTE): FP=193 (precision=0.2186)
- Decision Tree (Class Weight): FP=218 (precision=0.2351)
- Logistic Regression (SMOTE): FP=227 (precision=0.2357)
- Logistic Regression (Class Weight): FP=229 (precision=0.2417)
- Random Forest (Class Weight): FP=241 (precision=0.2325)

## 9. False-negative analysis

A false negative is a fraudulent test transaction the model missed. The test set has 96 fraud rows.

- Random Forest (Class Weight): FN=23, TP=73 (recall=0.7604)
- Logistic Regression (Class Weight): FN=23, TP=73 (recall=0.7604)
- Logistic Regression (SMOTE): FN=26, TP=70 (recall=0.7292)
- Decision Tree (Class Weight): FN=29, TP=67 (recall=0.6979)
- XGBoost (Class Weight): FN=34, TP=62 (recall=0.6458)
- Random Forest (SMOTE): FN=41, TP=55 (recall=0.5729)
- Decision Tree (SMOTE): FN=42, TP=54 (recall=0.5625)
- XGBoost (SMOTE): FN=87, TP=9 (recall=0.0938)

## 10. Recommended model

**Selected candidate:** Logistic Regression — Class Weight

- Artifact: `models/baseline/logistic_regression_class_weight.joblib`
- Accuracy=0.7480, Precision=0.2417, Recall=0.7604, F1=0.3668
- ROC-AUC=0.7646, PR-AUC=0.2620
- TN=675, FP=229, FN=23, TP=73

On this test set the selected candidate caught 73 of 96 frauds (true positives) and missed 23 frauds (false negatives: fraudulent transactions the model labeled legitimate). It correctly cleared 675 legitimate transactions and incorrectly flagged 229 legitimate transactions as fraud (false positives). Those 229 false positives would create review workload; the 23 false negatives are missed fraud loss.

## 11. Reason for recommendation

Logistic Regression (Class Weight) is the selected candidate because it has the strongest practical F1 (0.3668) at the default threshold, with precision 0.2417 and recall 0.7604. On the 1,000-row test set it yields TP=73, FN=23, FP=229, TN=675. PR-AUC=0.2620 and ROC-AUC=0.7646. The Always Legitimate dummy reaches accuracy 0.9040 with recall 0, which is why accuracy was not used to pick a winner. Highest PR-AUC is Logistic Regression (SMOTE) at 0.2681. If two models were within 0.01 F1, the simpler family / class-weight pipeline was preferred. Threshold calibration is deferred to Phase 7.

Dummy Always Legitimate is shown only to illustrate the accuracy trap. It is not a trained rival.

## 12. Limitations

- Test size is 1,000 rows / 96 fraud cases, so metric variance is non-trivial.
- Default threshold only; Phase 7 may change operating point.
- No probability calibration yet.
- SMOTE vs class-weight is a training choice already locked in Phase 5 artifacts.
- This candidate is not production-ready.
