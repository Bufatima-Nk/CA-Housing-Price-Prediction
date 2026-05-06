# California Housing Price Prediction — Ridge Regression

> End-to-end regression pipeline predicting California median house values on 20,640 blocks. Applied feature engineering to resolve multicollinearity, Ridge regularization, and a scikit-learn Pipeline — achieving **R² = 0.658** and reducing MAE by 48.1% vs. baseline.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20Regression-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-20%2C640%20blocks-purple)
![R²](https://img.shields.io/badge/R²-0.658-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Results

### Model Performance

| Metric | Baseline | Ridge Regression | Improvement |
|--------|:--------:|:----------------:|:-----------:|
| MAE | $91,364 | **$47,415** | **48.1% ↓** |
| RMSE | — | $65,806 | — |
| R² | 0.00 | **0.658** | — |

The Ridge regression model explains **65.8% of variance** in California housing prices, reducing mean prediction error from $91,364 to **$47,415** per block.

### Top Feature Importances (Ridge Coefficients, absolute value in USD)

| Rank | Feature | Importance | Insight |
|------|---------|:----------:|---------|
| 1 | `bedrooms_per_room` | $375,914 | Engineered — room quality proxy; fewer bedrooms per room = more spacious = higher value |
| 2 | `ocean_proximity` (Island) | $106,630 | Island properties command the highest premium |
| 3 | `ocean_proximity` (Near Bay) | $50,052 | Bay proximity adds ~$50K vs. inland |
| 4 | `median_income` | $43,431 | Strong positive driver — wealthier neighborhoods = higher prices |
| 5 | `population_per_household` | $31,116 | Engineered — household density proxy |
| 6 | `longitude` | $27,903 | West coast premium |
| 7 | `latitude` | $26,923 | Northern vs. Southern CA price gradient |

**Key insight:** The two most important features are both **engineered features** (`bedrooms_per_room`, `population_per_household`) — not raw columns from the original dataset. This validates the feature engineering step.

---

## Feature Engineering

Three new features were created to address multicollinearity and improve predictive power:

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `rooms_per_household` | `total_rooms / households` | Per-unit size; more informative than raw room count |
| `population_per_household` | `population / households` | Household density — overcrowding signal |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Room quality proxy — top predictor in final model |

**Multicollinearity resolution:** `total_bedrooms`, `households`, and `population` were dropped after correlation analysis showed high pairwise correlation (confirmed via heatmap). Retaining them would inflate Ridge coefficients and reduce interpretability.

---

## Dataset

- **Source:** California Housing Prices (1990 Census) — [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Records:** 20,640 census block groups
- **Target:** `median_house_value` (USD, range: $14,999 – $500,001)
- **Missing values:** 207 rows in `total_bedrooms` (1.0%) — imputed with `SimpleImputer`
- **Train/test split:** 80% / 20% (`random_state=42`)
- **Outlier treatment:** `total_rooms` trimmed at 10th–90th percentile

### Key Statistics

| Feature | Mean | Note |
|---------|------|------|
| `median_house_value` | $206,856 | Right-skewed; few high-value properties pull mean up |
| `median_income` | $38,707 | In tens of thousands USD |
| `housing_median_age` | 28.6 years | Range: 1–52 years |
| `ocean_proximity` dominant | `<1H OCEAN` (44%) | 5 categories total |

---

## Approach

### Pipeline Architecture

```
Input Features (9 columns)
    ↓
OneHotEncoder         → encode ocean_proximity (5 categories → 4 dummies)
    ↓
SimpleImputer         → fill 207 missing values in bedrooms_per_room (mean strategy)
    ↓
Ridge(alpha=1.0)      → L2-regularized linear regression
    ↓
Output: predicted median_house_value (USD)
```

Using `sklearn.pipeline.make_pipeline` ensures preprocessing and model are fit together — no data leakage between train and test sets.

### Why Ridge over OLS?

Plain Linear Regression is sensitive to multicollinearity and outliers. Ridge adds an L2 penalty term (λ‖w‖²) that shrinks large coefficients toward zero, reducing variance at the cost of a small bias. Given the correlated geographic features (`latitude`, `longitude`) and engineered ratio features, Ridge is the appropriate choice.

---

## Exploratory Analysis Highlights

**Location drives price.** The Plotly scatter mapbox confirms a clear coastal premium — properties near San Francisco Bay and the Pacific coast cluster at the highest price ranges.

**Right-skewed distributions.** `total_rooms`, `total_bedrooms`, `population`, and `households` are all heavily right-skewed (mean >> median), indicating a few very large blocks pull averages up. Outlier trimming at the 10th/90th percentile on `total_rooms` was applied.

**Island premium is real but thin.** Only 5 island properties exist in the dataset — the highest median value category, but too sparse to generalize from.

**Income is the most interpretable predictor.** `median_income` has the strongest direct correlation with `median_house_value` (r ≈ 0.69) among raw features.

---

## Project Structure

```
CA-Housing-Price-Prediction/
│
├── Regression Analysis.ipynb     # Full pipeline: EDA → feature engineering → modeling
├── housing.csv.zip               # Dataset (California Housing Prices)
├── 3d-scatter-plot.png           # 3D visualization of house value by location
├── requirements.txt              # Dependencies
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Bufatima-Nk/CA-Housing-Price-Prediction
cd CA-Housing-Price-Prediction
pip install -r requirements.txt
jupyter notebook "Regression Analysis.ipynb"
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Modeling | scikit-learn (Ridge, LinearRegression, Pipeline) |
| Encoding | category_encoders (OneHotEncoder) |
| Imputation | scikit-learn SimpleImputer |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly Express (scatter_mapbox, scatter_3d) |

---

## Limitations & Future Work

- **R² = 0.658** — reasonable for a linear model on this dataset, but gradient boosting (XGBoost, LightGBM) would likely reach R² > 0.82 with the same features.
- **Price cap at $500,001.** The dataset truncates values at $500,001 — a known artifact that creates a spike at the maximum and affects model calibration in the upper tail.
- **Geographic features as raw coordinates.** Lat/lon are used directly; clustering (k-means on location) or district-level aggregates would extract more signal.
- **No cross-validation.** Adding `cross_val_score` would give more reliable MAE estimates than a single 80/20 split.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
