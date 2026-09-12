# 2-minute project explanation (spoken)

Memorize this as a natural talk, not a list of slides.

---

Good morning. My internship project is **Financial Fraud Detection and Risk Analytics using Machine Learning with an Interactive Streamlit Dashboard**.

The problem is simple: most payments are genuine, and only a small share are fraud. If a system always says “legitimate”, accuracy looks high, but it catches zero fraud. Our dataset has **5,000 transactions**, **482 frauds**, so about **9.64%** fraud.

I did not use customer IDs, transaction IDs, or the suspicious-keyword column as model features, because those can leak identity or an undocumented rule. I engineered **18 row-local features**, including an amount-to-average ratio and cyclical hour and weekday encodings. Preprocessing — RobustScaler and one-hot encoding — was fitted on the **training set only**.

I compared eight models: Logistic Regression, Decision Tree, Random Forest, and XGBoost, each with SMOTE and with class weighting. I did not pick the highest accuracy. XGBoost with SMOTE reached **88.20%** accuracy but only **9.38%** recall. The selected candidate was **Logistic Regression with class weight**, because it had the best F1 and **76.04%** recall on the untouched test set at the default 0.50 cut.

For production I calibrated probabilities with **sigmoid calibration** on training data, then chose a review **threshold of 0.10** from training out-of-fold scores so recall stayed at least 0.70. The official test result at that threshold is **accuracy 74.30%, precision 23.78%, recall 76.04%**. That means the model caught **73 of 96** test frauds and wrongly flagged **234** legitimate transactions. Those flags are for **human review**, not automatic blocking.

The dashboard has five pages. It **loads the frozen joblib file**. It does **not** retrain. A risk score is probability times 100, with bands Low, Medium, High, and Critical. The score is a **risk indicator**, not proof of fraud.

This is an internship prototype with honest limits: precision is modest, 23 frauds are still missed, and it is not a certified banking system. Thank you.
