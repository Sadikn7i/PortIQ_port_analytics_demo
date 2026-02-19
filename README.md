# GHIH Smart Port Analytics Demo
### Machine Learning for Port Operations Optimization — Djibouti

---

## Overview

This project demonstrates how machine learning can optimize port operations at **Doraleh Multipurpose Port, Djibouti**. It was built as a data science portfolio project simulating 2 years of realistic port operations data across 4 ML components.

The project covers the full data science workflow: data generation, exploratory analysis, feature engineering, model training, evaluation, and business impact estimation.

---

## Business Problem

Ports like Doraleh handle thousands of containers daily. Without predictive analytics, operations teams face:
- Unpredictable container volumes causing staffing shortages
- Containers sitting too long at port (high dwell time = high costs)
- Inefficient truck routing wasting fuel and time
- Manual fraud/anomaly detection that misses suspicious containers

**This project shows how ML solves each of these problems.**

---

## Project Structure

```
port_analytics_demo/
│
├── data/
│   └── generate_data.py        # Generates all datasets
│
├── notebooks/
│   ├── analysis_traffic.py     # Component 1: Traffic Forecasting
│   ├── analysis_dwell.py       # Component 2: Dwell Time Prediction
│   ├── analysis_routes.py      # Component 3: Route Optimization
│   └── analysis_anomaly.py     # Component 4: Anomaly Detection
│
├── results/
│   └── visualizations/         # All output charts (auto-generated)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/port_analytics_demo.git
cd port_analytics_demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the data
cd data
python generate_data.py
cd ..

# 4. Run each analysis component
cd notebooks
python analysis_traffic.py
python analysis_dwell.py
python analysis_routes.py
python analysis_anomaly.py
```

---

## Component 1: Traffic Forecasting

**Goal:** Predict daily container arrival volumes up to 30 days ahead.

**Dataset:** 731 days of simulated daily container arrivals (Jan 2023 – Dec 2024) with realistic weekly cycles, seasonal trends, and growth patterns.

**Approach:**
- Feature engineering: lag variables (1, 7, 14, 30 days), rolling averages, day/month/quarter features
- Model: Random Forest Regressor (100 trees)
- Train/test split: 80/20 chronological split

**Results:**

| Metric | Value |
|--------|-------|
| Prediction Accuracy | **94.9%** |
| Mean Absolute Error | 21 containers/day |
| R² Score | 0.870 |

**Key Findings:**
- Weekend traffic drops by **32.6%** vs weekdays
- Top predictors: last week's volume (lag_7), 2-week lag, and day of week
- Model can forecast 30 days ahead with high confidence

**Business Impact:**
- Optimized staffing: ~$127,750/year savings
- Better crane and storage planning: 30% improvement
- Estimated total: **$166,000+/year**

---

## Component 2: Dwell Time Prediction

**Goal:** Predict how long each container will stay at port before pickup.

**Dataset:** 5,000 containers with origin country, cargo type, customs status, truck availability, and weight.

**Approach:**
- Label encoding of categorical variables
- Compared 3 models: Linear Regression, Random Forest, Gradient Boosting
- Best model selected based on R² score

**Results:**

| Model | Accuracy | R² Score | MAE |
|-------|----------|----------|-----|
| Linear Regression | 62.5% | 0.112 | 26.1h |
| Random Forest | 81.4% | 0.795 | 13.0h |
| **Gradient Boosting** | **82.3%** | **0.820** | **12.3h** |

**Key Findings:**
- **Customs Status** is the #1 predictor (importance: 0.505)
- Fast customs clearance: avg **50.7 hours** dwell time
- Delayed customs: avg **141.8 hours** — nearly 3x longer
- Ethiopian cargo is fastest to clear (**47.8h avg**)
- UAE cargo takes longest (**90.6h avg**)

**Business Impact:**
- Storage optimization savings: ~$3,000,000/year
- Faster truck turnaround: 8 hours average reduction
- Estimated total: **$4,200,000+/year**

---

## Component 3: Route Optimization

**Goal:** Classify truck routes by efficiency and identify fuel/time savings.

**Dataset:** 1,000 truck routes across 6 Djibouti locations:
- Doraleh Port, DIFTZ, Damerjog Port, City Center, Ethiopia Border, Airport

**Approach:**
- KMeans clustering (k=3) to group routes into Efficient / Moderate / Inefficient
- Random Forest model to predict fuel cost
- Traffic impact analysis

**Results:**

| Route Type | Count | Avg Fuel Cost | Avg Travel Time |
|------------|-------|--------------|-----------------|
| Efficient | 660 (66%) | $17.60 | 28.8 min |
| Moderate | 236 (23.6%) | $86.33 | 120.8 min |
| Inefficient | 104 (10.4%) | $106.74 | 205.8 min |

**Key Findings:**
- High traffic increases travel time by **85.1%**
- High traffic increases fuel cost by **21%**
- Most common route: Airport ↔ City Center
- Inefficient routes cost **$89 more** per trip than efficient ones

**Business Impact:**
- Fuel savings from optimizing inefficient routes: ~$3,380,000/year
- CO2 reduction: ~2,255,950 kg/year
- Estimated total: **$4,060,000+/year**

---

## Component 4: Anomaly Detection

**Goal:** Automatically flag suspicious containers for customs investigation.

**Dataset:** 5,000 containers with 250 injected anomalies (5%) — unusually fast or slow dwell times indicating potential fraud or delays.

**Approach:**
- Isolation Forest (unsupervised ML, 200 estimators)
- StandardScaler for feature normalization
- PCA for 2D visualization
- Risk scoring system: 0–100 for each container

**Results:**

| Metric | Value |
|--------|-------|
| Overall Accuracy | **91.9%** |
| Containers Scanned | 5,000 |
| Anomalies Flagged | 250 |
| F1 Score | 0.192 |

**Sample High-Risk Alerts Generated:**

| Container | Origin | Dwell Time | Risk Score |
|-----------|--------|------------|------------|
| CNT000680 | Saudi Arabia | 206.3h | 90.7 |
| CNT001918 | Ethiopia | 288.2h | 86.1 |
| CNT000192 | China | 14.3h | 85.4 |
| CNT000691 | Saudi Arabia | 270.4h | 84.4 |

**Key Findings:**
- Saudi Arabia origin has the highest anomaly rate (**6.3%**)
- Anomalous containers average **120.4h** dwell vs 69.8h normal
- Both extremely fast AND extremely slow dwell times are flagged
- Risk score 0–100 allows operations team to prioritize reviews

**Business Impact:**
- ~21 suspicious containers flagged per day automatically
- Fraud prevention value: ~$7,200,000/year
- Reduced manual inspection workload: 60–70%
- Estimated total: **$10,800,000+/year**

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML Models | scikit-learn |
| Visualization | matplotlib, seaborn |
| Algorithms | Random Forest, Gradient Boosting, KMeans, Isolation Forest, PCA |

---

## Key Takeaways

- **Customs status** is the single biggest driver of how long a container stays at port
- **Weekend planning** matters — traffic drops 32.6% on weekends
- **10.4% of truck routes** are classified as inefficient and are costing significantly more in fuel
- **Automated anomaly detection** can scan thousands of containers instantly, flagging the riskiest ones for human review

---

## Author

Built by Dirir as a data science portfolio project demonstrating ML applications in port logistics and supply chain operations.

*Note: All data in this project is synthetically generated for demonstration purposes and does not represent real port operations data.*


Requirements:
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
scikit-learn>=1.3.0