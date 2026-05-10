# 🏠 Real Estate Investment Advisor

### Predicting Property Profitability & Future Value

## 📌 Problem Statement

Build a machine learning application that:

1. **Classifies** whether a property is a _"Good Investment"_ (Classification)
2. **Predicts** the estimated property price after 5 years (Regression)

---

## 🗂️ Project Structure

```
real_estate_advisor/
├── data/
│   ├── india_housing_prices.csv       ← Raw dataset (add here)
│   └── processed_housing.csv          ← Generated after preprocessing
├── models/
│   ├── classifier.pkl                 ← Trained classifier
│   └── regressor.pkl                  ← Trained regressor
├── scripts/
│   ├── 01_preprocessing.py            ← Data cleaning & feature engineering
│   ├── 02_eda.py                      ← 20 EDA questions with charts
│   └── 03_model_training.py           ← Train + MLflow tracking
├── app/
│   └── app.py                         ← Streamlit application
├── eda_outputs/                       ← EDA charts (auto-generated)
├── model_reports/                     ← Model evaluation plots
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
cd real_estate_advisor
python scripts/01_preprocessing.py
```

Outputs: `data/processed_housing.csv`

### 3. Run EDA (generates 20+ charts)

```bash
python scripts/02_eda.py data/processed_housing.csv
```

Charts saved in `eda_outputs/`

### 4. Train Models (with MLflow tracking)

```bash
python scripts/03_model_training.py data/processed_housing.csv
```

Models saved in `models/`  
To view MLflow UI: `mlflow ui` (open http://localhost:5000)

### 5. Launch Streamlit App

```bash
streamlit run app/app.py
```

---

## 📊 Dataset Features

| Feature                          | Description                           |
| -------------------------------- | ------------------------------------- |
| `State / City / Locality`        | Location hierarchy                    |
| `Property_Type`                  | Apartment, Villa, House, etc.         |
| `BHK`                            | Bedrooms-Hall-Kitchen count           |
| `Size_in_SqFt`                   | Property area                         |
| `Price_in_Lakhs`                 | Property price                        |
| `Price_per_SqFt`                 | Normalized price metric               |
| `Year_Built`                     | Construction year                     |
| `Furnished_Status`               | Unfurnished / Semi / Fully            |
| `Nearby_Schools`                 | School proximity                      |
| `Nearby_Hospitals`               | Hospital proximity                    |
| `Public_Transport_Accessibility` | Low / Medium / High                   |
| `Parking_Space`                  | Number of spots                       |
| `Security`                       | Gated / CCTV / Guard                  |
| `Amenities`                      | Gym, Pool, Clubhouse, etc.            |
| `Owner_Type`                     | Individual / Builder / Agent          |
| `Availability_Status`            | Available / Under Construction / Sold |

### Engineered Features

| Feature                    | Formula                   |
| -------------------------- | ------------------------- |
| `Age_of_Property`          | `2024 - Year_Built`       |
| `Price_per_SqFt`           | `Price × 1e5 / Size`      |
| `School_Density_Score`     | Normalized school count   |
| `Hospital_Proximity_Score` | Normalized hospital count |
| `Appreciation_Rate`        | `Transport_Score / Age`   |
| `Amenity_Count`            | Count of listed amenities |
| `Transport_Score`          | Low=1, Medium=2, High=3   |

---

## 🎯 Target Variables

| Target                | Type         | Definition                                                                             |
| --------------------- | ------------ | -------------------------------------------------------------------------------------- |
| `Good_Investment`     | Binary (0/1) | `appreciation_rate > median` AND `price_per_sqft < 75th pct` AND `transport_score ≥ 2` |
| `Price_After_5_Years` | Continuous   | `Price × (1 + annual_rate)^5`                                                          |

---

## 🤖 Models Used

### Classification — Good Investment

| Model         | Metric             |
| ------------- | ------------------ |
| Random Forest | Accuracy + AUC-ROC |
| XGBoost       | Accuracy + AUC-ROC |

### Regression — Price After 5 Years

| Model         | Metric        |
| ------------- | ------------- |
| Random Forest | RMSE, MAE, R² |
| XGBoost       | RMSE, MAE, R² |

---

## 📈 EDA Questions Covered (All 20)

**Price & Size (Q1–Q5)**

- Distribution of prices & sizes
- Price per SqFt by property type
- Size vs Price relationship
- Outlier detection

**Location Analysis (Q6–Q10)**

- Avg price per SqFt by state
- Avg price by city
- Median age by locality
- BHK distribution across cities
- Price trends in top localities

**Feature Relationships (Q11–Q15)**

- Correlation heatmap
- Schools & Hospitals vs Price
- Price by furnished status
- Price by facing direction

**Investment Analysis (Q16–Q20)**

- Owner type distribution
- Availability status breakdown
- Parking vs Price
- Amenities vs Price
- Transport accessibility vs Investment

---

## 🖥️ Streamlit App Features

- **🔮 Predict** — Input form → Classification + Regression results + Confidence gauge
- **📊 EDA Dashboard** — Interactive Plotly charts across 4 tabs
- **🔍 Explore Data** — Filter by city, type, BHK, price, size + download CSV
- **ℹ️ About** — Project documentation

---

## 📦 Deliverables

- [x] Cleaned & processed dataset (CSV)
- [x] EDA scripts with 20+ visualizations
- [x] Model training scripts (Classification + Regression)
- [x] MLflow experiment tracking integration
- [x] Streamlit application with predictions + insights
- [x] Project documentation (this README)

---

## 🏷️ Technical Tags

`Python` `Pandas` `Scikit-learn` `XGBoost` `Random Forest`
`Regression` `Classification` `Streamlit` `MLflow` `Plotly`
`Real Estate Analytics` `Feature Engineering` `EDA`
