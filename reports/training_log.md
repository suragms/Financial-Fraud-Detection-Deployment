# Phase 5 Training Log

Candidate baseline pipelines were trained. **No model is declared best.**
Formal comparison belongs to Phase 6. No test-set metrics are reported here.

## Dataset

- File: `data/raw/financial_fraud_detection_dataset.csv`
- Rows after feature engineering: 5000
- Target: `Fraudulent`
- Feature count: 18

## Split

- test_size: 0.2
- stratify: y
- random_state: 42
- Train rows: 4000
- Test rows: 1000
- Train fraud count / rate: 386 / 9.65%
- Test fraud count / rate: 96 / 9.60%
- Overlapping Transaction_IDs: 0

The test fold is not resampled and is not used for fitting or threshold tuning.

## Feature groups

Imported from `src.features.feature_engineering` (single source of truth).

- Numeric: `Transaction_Amount`, `Average_Spend`, `Previous_Transactions`, `Account_Age_Days`, `amount_to_average_ratio`, `transaction_hour`, `transaction_day_of_week`, `transaction_month`, `transaction_year`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `Is_International`
- Categorical: `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`
- Excluded: `Transaction_ID`, `Customer_ID`, `Suspicious_Keyword`, `Fraudulent`, `Transaction_Date`

## Preprocessing

- `ColumnTransformer`
- Numeric: `RobustScaler` fit on **train only**
- Categorical: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fit on **train only**
- Dense one-hot is intentional: SMOTE cannot use a sparse matrix, and 5,000 rows stay small after encoding.

## Imbalance strategy

### Track A — SMOTE

`imblearn.pipeline.Pipeline`: preprocess -> SMOTE -> model. SMOTE runs only during `fit` on transformed training rows.

### Track B — class weight

No SMOTE. Logistic Regression / Decision Tree / Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight=9.3627` computed from the training fold only.

## Models trained

Conservative defaults, `random_state=42`. No grid search.

- Logistic Regression (`max_iter=2000`)
- Decision Tree (`max_depth=6`, `min_samples_leaf=20`)
- Random Forest (`n_estimators=100`, `max_depth=8`, `min_samples_leaf=10`)
- XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)

## Artifact locations

- `models/baseline/logistic_regression_smote.joblib`
- `models/baseline/decision_tree_smote.joblib`
- `models/baseline/random_forest_smote.joblib`
- `models/baseline/xgboost_smote.joblib`
- `models/baseline/logistic_regression_class_weight.joblib`
- `models/baseline/decision_tree_class_weight.joblib`
- `models/baseline/random_forest_class_weight.joblib`
- `models/baseline/xgboost_class_weight.joblib`
- `models/baseline/training_metadata.json`

## Validation checks

- X_test and y_test snapshots match after fitting.
- Train/test Transaction_IDs do not overlap.
- Each saved pipeline was reloaded with joblib and `predict_proba` was called on a small X_test sample (labels unused).

Reload checks:

- `logistic_regression_smote.joblib`: sample=8, probability_shape=[8, 2]
- `decision_tree_smote.joblib`: sample=8, probability_shape=[8, 2]
- `random_forest_smote.joblib`: sample=8, probability_shape=[8, 2]
- `xgboost_smote.joblib`: sample=8, probability_shape=[8, 2]
- `logistic_regression_class_weight.joblib`: sample=8, probability_shape=[8, 2]
- `decision_tree_class_weight.joblib`: sample=8, probability_shape=[8, 2]
- `random_forest_class_weight.joblib`: sample=8, probability_shape=[8, 2]
- `xgboost_class_weight.joblib`: sample=8, probability_shape=[8, 2]

Phase 6 will compare models on the untouched test set.
