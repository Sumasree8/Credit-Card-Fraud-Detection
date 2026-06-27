# 💳 Credit Card Fraud Detection

> An end-to-end machine learning pipeline that detects fraudulent credit card transactions on a highly imbalanced dataset of **1.85M+ transactions**, using SMOTE for class balancing and a Random Forest classifier.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/imbalanced--learn-SMOTE-green" alt="imbalanced-learn">
  <img src="https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Status-Complete-success" alt="Status">
</p>

---

## 📌 Overview

Credit card fraud costs the financial industry billions every year, yet fraudulent transactions make up a tiny fraction of all activity — making them genuinely hard to catch. This project builds a complete supervised-learning workflow that:

- Ingests **1.29M training** and **555K test** transactions (23 raw features)
- Engineers time-based behavioural features from raw transaction timestamps
- Tackles a severe **0.58% fraud class imbalance** using **SMOTE** oversampling
- Trains and evaluates a **Random Forest** classifier with fraud-focused metrics

The result is a model that correctly flags fraud with **89% precision** while keeping false alarms extremely low on a real-world-scale dataset.

---

## 📊 Results

Evaluated on a held-out test set of **555,719 transactions** (2,145 actual fraud cases):

| Metric | Fraud Class (1) | Notes |
|---|---|---|
| **Precision** | **0.89** | 9 out of 10 flagged transactions are truly fraudulent |
| **Recall** | **0.60** | Catches 60% of all fraud |
| **F1-Score** | **0.72** | Strong balance on a 0.58%-positive class |
| **ROC-AUC** | **0.80** | Solid class separability |
| **Accuracy** | **~1.00** | (Expected — driven by the majority class) |

**Confusion Matrix**

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 553,408 ✅ | 166 (false alarms) |
| **Actual Fraud** | 855 (missed) | 1,290 ✅ |

> 💡 Only **166 false positives out of ~553K legitimate transactions** — meaning the model rarely inconveniences genuine customers.

---

## 🧠 Approach

\`\`\`
Raw Data (1.85M txns, 23 cols)
        │
        ▼
  Data Cleaning ──────────►  Drop nulls + high-cardinality PII columns
        │                    (names, address, dob, job, trans_num…)
        ▼
  Feature Engineering ─────►  Extract hour / day / month from timestamps
        │                    Label-encode categorical features
        ▼
  Scaling ─────────────────►  StandardScaler
        │
        ▼
  Class Balancing ─────────►  SMOTE (7,506 → 1,289,169 fraud samples)
        │
        ▼
  Modeling ────────────────►  RandomForestClassifier (100 trees)
        │
        ▼
  Evaluation ──────────────►  Confusion Matrix · Precision/Recall · ROC-AUC
\`\`\`

### Key design decisions
- **SMOTE applied only on training data** — prevents data leakage into the test set, a common mistake in imbalanced-learning projects.
- **PII / high-cardinality columns dropped** — focuses the model on generalizable signal rather than memorizing identities.
- **Temporal feature extraction** — transaction *hour/day/month* often carry strong fraud signal that raw timestamps hide.

---

## 🔍 Exploratory Data Analysis

The notebook includes visual analysis to motivate the modeling choices:

- **Fraud vs. Non-Fraud distribution** — exposing the extreme class imbalance
- **Transaction amount by class** — comparing amount ranges of fraudulent vs. legitimate transactions
- **Feature correlation heatmap** — identifying redundant/related numeric features

---

## 🗂️ Dataset

This project uses the **Sparkov simulated credit card transactions** dataset (`fraudTrain.csv` / `fraudTest.csv`).

| Split | Rows | Columns |
|---|---|---|
| Train | 1,296,675 | 23 |
| Test | 555,719 | 23 |

📥 Available on Kaggle: [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

> The CSV files are **not included** in this repo due to size. Download them from Kaggle and update the file paths in the notebook.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML & Preprocessing | scikit-learn (Random Forest, StandardScaler, LabelEncoder) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook / Google Colab |

---

## 🚀 Getting Started

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/Sumasree8/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
\`\`\`

### 2. Install dependencies
\`\`\`bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
\`\`\`

### 3. Add the data
Download `fraudTrain.csv` and `fraudTest.csv` from Kaggle and place them in the project (or update the paths in the notebook).

### 4. Run the notebook
\`\`\`bash
jupyter notebook fraud_detection.ipynb
\`\`\`
Or open it directly in **Google Colab** and mount the dataset from Drive.

---

## 📁 Repository Structure

\`\`\`
Credit-Card-Fraud-Detection/
├── fraud_detection.ipynb   # End-to-end pipeline: EDA → preprocessing → SMOTE → model → evaluation
└── README.md
\`\`\`

---

## 🔮 Future Improvements

- [ ] Compare against **XGBoost / LightGBM** and gradient-boosted baselines
- [ ] **Hyperparameter tuning** (GridSearchCV / Optuna) to lift recall on fraud
- [ ] Threshold optimization using the **Precision-Recall curve** instead of the default 0.5
- [ ] **Feature importance** analysis to interpret fraud drivers
- [ ] Package the model behind a **Flask/FastAPI** endpoint for real-time scoring

---

## 👩‍💻 Author

**Sumasree**
🔗 [GitHub](https://github.com/Sumasree8)

---

⭐ *If you found this project useful or interesting, consider giving it a star!*
