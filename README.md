# Customer Analytics — Segmentation, CLV & Churn Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.90-success)
![Models](https://img.shields.io/badge/Models-5%20Compared-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)

## Live Demo
🚀 **[Launch Dashboard →](YOUR_STREAMLIT_URL_HERE)**

---

## Business Problem
A UK-based online retailer needed to:
1. Understand **who their customers are** (segmentation)
2. Predict **which customers will churn** (retention)
3. Quantify **how much revenue is at risk** (CLV)

**Result:** Identified £484,992 in CLV at risk from 109 high-value 
churning customers, enabling targeted retention with 5–8x ROI.

---

## Project Architecture
1,067,371 raw transactions

↓

Data Cleaning (73% retained → 779,425 rows)

↓

Feature Engineering (13 RFM + behavioral features)

↓

┌────────────────┬─────────────────┬──────────────┐

│  Segmentation  │ Churn Prediction│ CLV Prediction│

│  K-Means K=4   │ CatBoost 0.90   │ XGBoost 0.90 │

└────────────────┴─────────────────┴──────────────┘

↓

SHAP Explainability + MLflow Tracking

↓

Streamlit Dashboard (5 pages)

---

## Key Results

| Metric | Value |
|--------|-------|
| Customers Analysed | 5,878 |
| Transactions Processed | 779,425 |
| Churn Model ROC-AUC | **0.90** |
| CLV Model Accuracy | **90%** |
| Revenue at Risk | **£484,992** |
| High-Risk Customers | 109 |
| Models Compared | 5 |

---

## Customer Segments

| Segment | Customers | Revenue Share | Churn Rate |
|---------|-----------|---------------|------------|
| Champions | 1,520 | **78.4%** | 14% |
| Loyal Developing | 2,031 | 13.3% | 43% |
| At Risk | 700 | 5.8% | 73% |
| One-Time Lost | 1,627 | 2.5% | 85% |

---

## Tech Stack

**Data & ML:** Python, Pandas, NumPy, Scikit-learn,
XGBoost, LightGBM, CatBoost, SHAP

**Visualisation:** Matplotlib, Seaborn, Plotly

**MLOps:** MLflow (experiment tracking, model versioning)

**Deployment:** Streamlit Cloud

---

## Project Structure
customer_analytics/

├── data/

│   ├── raw/                    # Original dataset

│   └── processed/              # Cleaned + engineered data

├── notebooks/

│   ├── 01_data_cleaning.ipynb

│   ├── 02_eda.ipynb

│   ├── 03_feature_engineering.ipynb

│   ├── 04_segmentation.ipynb

│   ├── 05_churn_prediction.ipynb

│   ├── 06_explainability.ipynb

│   ├── 07_mlflow_tracking.ipynb

│   └── 08_clv_prediction.ipynb

├── models/saved/               # Trained model files

├── reports/figures/            # All generated plots

├── src/                        # Modular source code

├── app.py                      # Streamlit dashboard

├── requirements.txt

└── README.md

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/customer-analytics-ml
cd customer-analytics-ml
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Insights

- **Seasonality drives churn** — PreferredMonth was the #1
  churn predictor. Customers who only buy in Nov/Dec
  are the highest churn risk.
- **Tenure is the strongest retention signal** — customers
  active for 600+ days have only 14% churn rate vs
  85% for first-time buyers.
- **Top 23% of customers generate 80% of revenue** —
  even more concentrated than the classic 80/20 rule.
- **£484,992 CLV at risk** from just 109 high-value
  churning customers — average £4,449 per customer.

---

## Author
**[S.Arun Karthik]**
[LinkedIn](https://www.linkedin.com/in/arun-karthik-138a87255/) | [GitHub](https://github.com/Arunkarthik134630)
