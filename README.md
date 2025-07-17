# 🔍 Fraud Detection Using Machine Learning

This project involves building a machine learning model to proactively detect fraudulent financial transactions using real-world-style transaction data. The dataset contains 6.36 million records and 11 features, including transaction details, account balances, and labels for fraud.

---

## 📁 Dataset Overview

- **Source**: Simulated financial transaction data
- **Rows**: 6,362,620
- **Columns**: 11
- **Target variable**: `isFraud` (1 = fraud, 0 = legitimate)

### Key Columns:
- `type`: Transaction type (TRANSFER, CASH_OUT, etc.)
- `amount`: Transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: Originator account balances
- `oldbalanceDest`, `newbalanceDest`: Receiver account balances
- `isFraud`: Whether the transaction was fraudulent

---

## 📊 Exploratory Data Analysis

### ✅ Class Imbalance in `isFraud`
![Class Imbalance](images\fraud_vs_nonfraud.jpg)

---

### ✅ Fraud Distribution by Transaction Type
Only `TRANSFER` and `CASH_OUT` types are fraudulent.
![Fraud by Type](images\fraud_by_type.jpg)

---

### ✅ Hourly Fraudulent Transactions Over Time
This plot helps understand **when fraud typically occurs**, based on the `step` column (representing hourly steps in the simulation).

![Hourly Fraud](images\Hourly_Fraudulent_Transactions_Over_Time.jpg)

---

### ✅ Transaction Amount by Fraud Class
Fraudulent transactions usually have high amounts.
![Amount Boxplot](images\amount_vs_isfraud_boxplot.jpg)

---

### ✅ Correlation Heatmap of Features
![Correlation Heatmap](images\correlation_heatmap.jpg)


---

## 🧹 Data Cleaning & Feature Engineering

- Removed irrelevant columns: `nameOrig`, `nameDest`, `isFlaggedFraud`
- Handled missing and infinite values using `.fillna(0)` and `.replace()`
- Encoded categorical column `type` using `LabelEncoder`
- Added two new features:
  - `errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig`
  - `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`

---

## 🤖 Model Used

- `RandomForestClassifier` from `sklearn.ensemble`
- Trained on full dataset (6.36M records)
- Handled class imbalance using `class_weight='balanced'`

---

## 📈 Model Performance

| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | 100%      |
| Precision (fraud) | 100%      |
| Recall (fraud)    | 99.7%     |
| ROC AUC Score     | 0.9985    |

> ✅ Confusion Matrix showed only **6 false negatives** and **0 false positives**

---

## 🔑 Top Fraud Indicators

1. `type = TRANSFER or CASH_OUT`
2. High `amount` value
3. Sudden or inconsistent account balance changes (`errorBalanceOrig`, `errorBalanceDest`)
4. Accounts with low `oldbalanceOrg` sending large transfers

---

## 🛡️ Fraud Prevention Recommendations

- Flag transactions with:
  - Large amounts
  - Unusual balance patterns
  - Specific transaction types (`TRANSFER`, `CASH_OUT`)
- Implement real-time scoring using this model
- Create a feedback loop from confirmed fraud cases
- Monitor high-risk accounts with behavioral anomalies

---

## ✅ Future Work

- Try advanced models (XGBoost, LightGBM)
- Deploy using Streamlit or FastAPI
- Build a Power BI or Plotly dashboard for live fraud monitoring
- Automate model retraining using new flagged data

---

## 💻 Tools & Libraries

- Python 3.9
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- joblib (for saving model)

---

## 📦 How to Run

1. Clone this repo
2. Install dependencies  
   `pip install -r requirements.txt`
3. Place your dataset CSV in the root directory
4. Run `Fraud_detection.ipynb` or the script version
5. Load model:  
   ```python
   import joblib  
   model = joblib.load("fraud_detection_model.pkl")
