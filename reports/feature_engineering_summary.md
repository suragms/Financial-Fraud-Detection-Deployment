# Phase 4 Feature Engineering Summary

No models were trained in this phase. No SMOTE, scaling, or encoding was applied.
The raw CSV is not modified. Encoders and scalers belong in Phase 5 and must be fit on training data only.

## Raw columns

`Transaction_ID`, `Customer_ID`, `Transaction_Date`, `Transaction_Amount`, `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`, `Is_International`, `Previous_Transactions`, `Average_Spend`, `Account_Age_Days`, `Suspicious_Keyword`, `Fraudulent`

- Rows in: 5000
- Rows out: 5000
- Parsed date min: 2023-01-01 02:16:00
- Parsed date max: 2024-02-21 15:34:00

## Engineered columns

`amount_to_average_ratio`, `transaction_hour`, `transaction_day`, `transaction_day_of_week`, `transaction_month`, `transaction_year`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

`transaction_day` is derived but **not** included in `MODEL_FEATURES` (day-of-month was not an EDA association; month and day-of-week already cover calendar structure).

## Included model features

### Numeric

- `Transaction_Amount`
- `Average_Spend`
- `Previous_Transactions`
- `Account_Age_Days`
- `amount_to_average_ratio`
- `transaction_hour`
- `transaction_day_of_week`
- `transaction_month`
- `transaction_year`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `Is_International`

### Categorical (not one-hot encoded here)

- `Merchant_Category`
- `Payment_Method`
- `Device_Type`
- `Location`

Production `build_features()` returns 18 columns: 14 numeric + 4 categorical.

## Excluded features

- `Transaction_ID`
- `Customer_ID`
- `Suspicious_Keyword`
- `Fraudulent`
- `Transaction_Date`

## Reasons for exclusions

- `Transaction_ID`: unique row identifier. Useful as dashboard metadata only.
- `Customer_ID`:  high cardinality; median of one transaction per customer. Raw IDs do not generalise. Naive full-data aggregates (`customer_total_transactions`, `customer_average_amount`, `customer_fraud_rate`) would leak future or target information.
- `Suspicious_Keyword`: strong observed association with fraud, but generation time is undocumented. Held out of production X pending provenance review.
- `Fraudulent`: target. Never included in X.
- `Transaction_Date`: raw timestamp string is replaced by hour/dow/month/year and cyclical encodings.

## Leakage protections

- Features are row-local. No group-by on the full file, no target encoding, no fraud rates, no customer history from all rows.
- The target is not read by `build_features` / `engineer_feature_frame`.
- No scaler or encoder is fitted. Categorical values stay as labels for a future train-only `ColumnTransformer`.
- Input DataFrame is copied by construction; callers receive a new frame.
- Invalid dates raise instead of becoming silent missing values.
- `validate_feature_matrix` rejects target, IDs, keyword, raw date, inf, and unexpected NaNs.

## Customer_ID decision

EXCLUDE raw `Customer_ID` from the production feature matrix.

Customer-level historical aggregates are a **future enhancement** and must be built from training-time history only, never from the full dataset or from the label.

## Suspicious_Keyword decision

EXCLUDE from production ML features. Keep the column in the raw file for EDA/dashboard analysis and provenance investigation. Do not delete it from `data/raw/`.

## Date/time transformations

Parsed with format `%d-%m-%Y %H:%M`. Derived: `transaction_hour` (0-23), `transaction_day` (debug only), `transaction_day_of_week` (Monday=0), `transaction_month`, `transaction_year`.

Cyclical encodings: `hour_sin`/`hour_cos` = sin/cos(2π hour / 24); `dow_sin`/`dow_cos` = sin/cos(2π dow / 7). These let linear models treat 23:00 as close to 00:00.

Month and year are kept as numeric fields. They do **not** represent proven seasonality; the observed window is limited and the final month is partial.

## Amount / average ratio

- Mean: 0.5889
- Median: 0.2411
- Min: 0.0000
- Max: 28.3633
- 99th percentile: 6.0013
- Non-finite values: 0
- Rows with Average_Spend <= 0 in this file: 0

If `Average_Spend <= 0`, the ratio is defined as `0.0` so serving code never emits inf/NaN. Extreme ratios are retained (not dropped). No `log1p` transform is applied yet: the ratio is interpretable, and scaling/log choices belong in Phase 5 if training evidence supports them.

No rule of the form `amount > 200000 = fraud` is implemented. The amount feature is the raw value plus the ratio only.

## Future feature ideas

- Train-only customer history (prior count, prior mean amount) using timestamps strictly before the current row.
- `log1p(Transaction_Amount)` only if Phase 5 diagnostics show it helps a linear model.
- Revisit `Suspicious_Keyword` if documentation proves it is known before the decision.
- `transaction_day` cyclical encoding if a later EDA slice shows a month-day pattern.

ML training and dashboard implementation are planned for later phases.
