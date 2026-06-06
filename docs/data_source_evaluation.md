# Data Source

## Active AQI Data

The AQI page uses `data/central_taiwan_aqi_sample.csv`.

- Rows: 82,034
- Scope: Taichung, Changhua, and Nantou air-quality records
- Fields: station, county, publish time, Air Quality Index (AQI), PM2.5, PM10, O3, NO2, CO, SO2, wind speed, wind direction, longitude, latitude, and site id
- Purpose: support a practical linear-regression case with enough rows for residual ranking and model comparison

The 24-row demo fallback has been removed. If `central_taiwan_aqi_sample.csv` is unavailable, the project should use another sufficiently large public dataset or change the research case.

## Lab Data

The Lab page does not use teacher-provided data, Kaggle data, MOENV data, or any CSV file.

Lab data is generated inside `app.py` from:

```text
y = a*x + b + noise
```

The sidebar controls are the data-generation source for Lab:

- sample size `n`
- slope `a`
- intercept `b`
- noise variance
- random seed

This separation is intentional. Lab teaches the mechanics of linear regression; AQI applies the same modeling workflow to real air-quality records.

## Reference Sources

Kaggle `Taiwan Air Quality Index Data 2016~2024` is kept as a reproducible historical reference:
https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024

MOENV `AQX_P_488` is the preferred official historical AQI source:
https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488

MOENV `AQX_P_432` is the real-time hourly AQI reference:
https://data.moenv.gov.tw/dataset/detail/aqx_p_432

## Design Boundary

Sidebar controls behave differently by page:

- Lab controls change the synthetic data itself.
- AQI controls filter the active CSV and change model settings.
- Lab parameters such as slope, intercept, noise variance, and seed do not modify AQI records.
