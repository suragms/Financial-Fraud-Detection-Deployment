# Phase 3 EDA Summary

All figures below are calculated from `data/raw/financial_fraud_detection_dataset.csv`.
The raw CSV is not modified. Temporary datetime and ratio columns exist only in memory.

## 1. Dataset overview

- Rows: 5000
- Columns: 14
- Missing values: 0
- Duplicate rows: 0
- Duplicate Transaction_ID: 0
- Parsed date range: 2023-01-01 02:16:00 to 2024-02-21 15:34:00
- Required columns match the Phase 2 schema.

## 2. Target distribution

- Fraudulent transactions: 482 (9.64%)
- Legitimate transactions: 4518 (90.36%)

Class imbalance matters for later evaluation: a model that predicts all transactions as legitimate would already be about 90.36% accurate. Accuracy alone is not an appropriate selection metric. Later phases should emphasise recall, precision, F1, ROC-AUC, and PR-AUC on an untouched test set.

## 3. Numerical feature findings

- Transaction_Amount: overall mean 79.18, median 55.45, max 653.80. Fraud mean 81.99 vs legitimate mean 78.88; medians are nearly identical (55.43 vs 55.45). Correlation with the target is 0.0116. Amount alone shows limited separation.
- Previous_Transactions: fraud mean 96.86 vs legitimate 99.69; correlation -0.0146.
- Average_Spend: fraud and legitimate means are nearly the same (258.86 vs 258.33); correlation 0.0011.
- Account_Age_Days: fraud mean 1026.1 vs legitimate 1009.6; correlation 0.0086. EDA age groups stay near the overall fraud rate.
- amount_to_average_ratio (EDA-only): mean 0.685 for fraud vs 0.579 for legitimate. Zero Average_Spend rows: 0. This ratio is a Phase 4 candidate, not a saved column.

Outliers were inspected and **not** removed. The largest amounts remain in the file because fraud can occur in the tail. No amount threshold rule is justified.

## 4. Categorical feature findings

- Is_International: domestic fraud rate 7.00% (319/4559); international fraud rate 36.96% (163/441). This is the strongest non-keyword association observed.
- Merchant_Category: highest observed rate Entertainment 11.43% (n=630); lowest Electronics 7.54% (n=650). Category counts are similar (about 590-680), so rates are comparable.
- Payment_Method: highest UPI 10.41%; lowest NetBanking 8.20%. Differences are modest.
- Device_Type: Desktop 10.24% (n=1679), POS 9.50% (n=1673), Mobile 9.16% (n=1648).
- Location: highest Chennai 10.53%; lowest Delhi 8.58%. These are fraud rates by location, not evidence that a city causes fraud.

## 5. Time findings

- Coverage is 2023-01-01 through 2024-02-21 (4380 rows in 2023 and 620 in 2024). February 2024 is a partial month. This limited window is a limitation; seasonality is not claimed.
- Overnight hours 00:00-05:00 have a higher observed fraud rate (range 20.37% to 30.32%, about 178-236 transactions per hour) than hours 07:00-23:00 (range 2.28% to 5.58%). Hour is a useful Phase 4 candidate.
- Month with the highest observed rate: 2023-10 (12.72%, n=401). Month with the lowest: 2023-05 (7.97%, n=389).

## 6. Suspicious_Keyword leakage assessment

- Keyword = No: 7.57% (359/4740).
- Keyword = Yes: 47.31% (123/260).
- Domestic + No: 4.98% (n=4321).
- Domestic + Yes: 43.70% (n=238).
- International + No: 34.37% (n=419).
- International + Yes: 86.36% (n=22).

Suspicious_Keyword is strongly associated with the target. The dataset documentation does not prove it is a post-decision field, so it is not labelled as definite leakage. It **requires a provenance/timing review before inclusion in the production model** and is not used as a production ML feature in this phase.

## 7. Customer_ID assessment

- Unique customers: 3847
- Transactions per customer: mean 1.30, median 1, max 5
- Customers with more than one transaction: 952
- Customers with at least one fraud label: 472

Do not one-hot encode raw Customer_ID (high cardinality). Customer-level aggregates could be considered in Phase 4 only if they are computed from training data with no future-data leakage.

## 8. Correlation observations

- Is_International vs Fraudulent: 0.2879 (strongest numeric/binary correlation).
- Transaction_Amount vs Fraudulent: 0.0116.
- Previous_Transactions vs Fraudulent: -0.0146.
- Average_Spend vs Fraudulent: 0.0011.
- Account_Age_Days vs Fraudulent: 0.0086.

Correlation is not causation and is not used as the only feature-selection rule. Categorical rates and hour-of-day patterns add information that a Pearson matrix does not capture.

## 9. Candidate features

**KEEP:** Transaction_Amount, Previous_Transactions, Average_Spend, Account_Age_Days, Is_International, Merchant_Category, Payment_Method, Device_Type, Location, Transaction_Date-derived fields.

**TRANSFORM:** amount_to_average_ratio, hour, day of week, month.

**EXCLUDE:** Transaction_ID, raw Customer_ID, Fraudulent (target only).

**INVESTIGATE:** Suspicious_Keyword (provenance/timing review).

These labels are EDA recommendations. They are not permanently enforced until Phase 4.

## 10. Limitations

- The file has 5,000 rows and 482 fraud cases, so rate estimates have sampling variability.
- Dates cover about 14 months with a partial final month; seasonal claims are not supported.
- Suspicious_Keyword timing is undocumented.
- Customer history is thin (median one transaction).
- This phase does not train or evaluate models, so no performance numbers are reported.

## 11. Key findings

1. The dataset has 5000 transactions and 482 fraud cases (9.64% fraud rate), with no missing values and no duplicate Transaction_IDs.
2. International transactions have a higher observed fraud rate (36.96%) than domestic transactions (7.00%).
3. Suspicious_Keyword = Yes is strongly associated with the target (47.31% vs 7.57% when No) and therefore requires provenance review.
4. Transaction amount alone shows limited separation: fraud and legitimate medians are 55.43 and 55.45.
5. Overnight hours (00:00-05:00) have higher observed fraud rates than daytime hours; hour is a useful derived feature candidate.
6. Merchant, payment, device, and location fraud rates vary modestly (roughly 8.58% to 11.43%) with comparable sample sizes.
7. Previous_Transactions, Average_Spend, and Account_Age_Days have near-zero correlations with the target (|r| < 0.02) but remain KEEP candidates for non-linear models.
8. The amount-to-average-spend ratio is slightly higher on average for fraud (0.685 vs 0.579) and is a TRANSFORM candidate.
9. Raw Customer_ID should not be one-hot encoded; any later customer aggregates must avoid future leakage.
10. No amount threshold such as amount > 200000 is supported: the maximum amount in this file is 653.80.
11. Pearson correlation understates categorical and time-of-day associations; Is_International is the exception among numeric/binary fields.

ML training and dashboard implementation are planned for later phases.
