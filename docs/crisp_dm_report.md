# CRISP-DM Report: Linear Regression Practice

## 1. Business Understanding

This project uses central Taiwan air-quality data to identify observations that do not follow the usual pollutant pattern. The practical goal is to help a non-technical user notice suspicious AQI readings faster.

## 2. Data Understanding

Current app data: bundled central Taiwan AQI sample

Bundled sample note: this is a compact classroom demo dataset with field logic aligned to Taiwan AQI open data. It keeps station, county, publish time, AQI, PM2.5, PM10, O3, NO2, CO, SO2 and wind-speed-like fields. It is useful for demonstrating numeric modeling, but it is not a live API pull.

Recommended reproducible dataset reference: Kaggle Taiwan Air Quality Index Data 2016~2024

Kaggle URL: https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024

Official replacement source: MOENV AQX_P_432

MOENV URL: https://data.moenv.gov.tw/dataset/detail/aqx_p_432

Source judgment: Kaggle is suitable for reproducible classroom and portfolio practice because it is already shaped for analysis. MOENV AQX_P_432 is closer to the source of truth and should be used for a future production-like version.

## 3. Data Preparation

- Normalize common AQI column names.
- Convert pollutant and weather fields into numeric values.
- Filter central Taiwan counties when county data exists.
- Drop rows missing the selected target or feature fields.
- Let users choose a target and feature set in the interface.

## 4. Modeling

The app stays within the linear-regression assignment scope but adds multiple linear model variants:

- OLS Linear Regression: the clearest baseline for explaining a linear relationship.
- Ridge Regression: keeps all features while reducing coefficient instability, useful when pollutant fields are correlated.
- Lasso Regression: can shrink weaker feature effects toward zero, useful for feature screening.
- ElasticNet Regression: balances Ridge and Lasso for a practical stability/simplification compromise.

## 5. Evaluation

The app reports R-squared, RMSE and MAE. It also ranks observations by absolute residual. Large residuals are the records where the model's guess differs most from the actual value.

## 6. Deployment

The result is packaged as a bilingual Streamlit app with light/dark themes, sample data download, source notes, plain-language explanations below the chart and a CRISP-DM report generated in the currently selected language. External upload is intentionally removed until a schema validator is defined.
