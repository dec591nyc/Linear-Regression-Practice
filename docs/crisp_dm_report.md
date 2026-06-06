# CRISP-DM Report: Linear Regression Practice

## 1. Business Understanding

This project uses central Taiwan air-quality data to identify observations that do not follow the usual pollutant pattern. The practical goal is to help a non-technical user notice suspicious AQI readings faster.

## 2. Data Understanding

Default source: Kaggle Taiwan Air Quality Index Data 2016~2024  
Default source URL: https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024

Official replacement source: MOENV AQX_P_432  
Official source URL: https://data.moenv.gov.tw/dataset/detail/aqx_p_432

The bundled sample keeps station, county, publish time, AQI, PM2.5, PM10, O3, NO2, CO, SO2 and wind-speed-like numeric fields. The default scope focuses on Taichung and Changhua.

## 3. Data Preparation

- Normalize common AQI column names.
- Convert pollutant and weather fields into numeric values.
- Filter central Taiwan counties when county data exists.
- Drop rows missing the selected target or feature fields.
- Let users choose a target and feature set in the interface.

## 4. Modeling

The app trains a simple linear regression baseline. It predicts a numeric target, such as AQI, from selected pollutant fields.

## 5. Evaluation

The app reports R-squared, RMSE and MAE. It also ranks observations by absolute residual. Large residuals are the records where the model's guess differs most from the actual value.

## 6. Deployment

The result is packaged as a bilingual Streamlit app with light/dark themes, sample data download, CSV upload, source notes and plain-language explanations below the chart.
