# CRISP-DM Report: Linear Regression Practice

## 1. Business Understanding

This project uses Air Quality Index (AQI) and pollutant fields to build a linear-regression baseline. The goal is not to replace official air-quality forecasts. The goal is to help users notice records where a simple model cannot explain the observed air-quality value well.

## 2. Data Understanding

Active AQI dataset:

- File: `data/central_taiwan_aqi_sample.csv`
- Rows: 82,034
- Scope: Taichung, Changhua, and Nantou air-quality records
- Fields: station, county, publish time, Air Quality Index (AQI), PM2.5, PM10, O3, NO2, CO, SO2, wind speed, wind direction, longitude, latitude, and site id

Reference sources:

- Kaggle Taiwan Air Quality Index Data 2016~2024: https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024
- MOENV AQX_P_488 historical AQI source: https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488
- MOENV AQX_P_432 realtime hourly AQI reference: https://data.moenv.gov.tw/dataset/detail/aqx_p_432

Lab source boundary:

The Lab page does not use teacher-provided data, Kaggle data, MOENV data, or any CSV file. It generates synthetic data in memory inside `app.py` from `y = a*x + b + noise`. The Lab sidebar parameters are the source of that generated dataset.

## 3. Data Preparation

- Normalize AQI, pollutant, wind, and time column names.
- Convert model-ready fields into numeric values.
- Filter Central Taiwan, Taichung, Changhua, or Nantou through the sidebar.
- Drop rows missing the selected target or feature fields.
- Keep source, file name, row count, and scope visible so Lab and AQI are not confused.

## 4. Modeling

The app stays within the linear-regression assignment scope and supports four model variants:

- OLS Linear Regression: the clearest baseline for explaining a linear relationship.
- Ridge Regression: keeps all selected features while reducing coefficient instability when inputs are correlated.
- Lasso Regression: can shrink weaker feature effects toward zero for feature screening.
- ElasticNet Regression: balances Ridge and Lasso for stability and simplification.

## 5. Evaluation

The app reports R-squared, RMSE, and MAE. It also ranks observations by absolute residual.

For a non-technical reader, residual means the gap between the real value and the model's predicted value. The top residual rows are not automatically wrong. They are the records most worth checking first because the baseline model was most surprised by them.

## 6. Deployment

The result is packaged as a bilingual Streamlit app with light/dark themes, icon-based page navigation, dynamic sidebar controls, active CSV download, plain-language interpretation, and a CRISP-DM report generated in the currently selected language.
