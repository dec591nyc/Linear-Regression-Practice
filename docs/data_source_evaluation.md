# Data Source Evaluation

## Recommended classroom source

Kaggle `Taiwan Air Quality Index Data 2016~2024` is the preferred reproducible source for the air-quality case because it already packages Taiwan AQI records for machine-learning practice. The dataset theme also maps naturally to Taichung and Changhua through station or county fields.

## Official replacement source

The official replacement source is Ministry of Environment `AQX_P_432`, the hourly station-level AQI open dataset. Its fields include station name, county, AQI, pollutant, status, SO2, CO, O3, PM10, PM2.5, NO2, wind speed, wind direction, publish time, longitude, latitude, and site id.

## Why AQI fits this assignment

- The target is numeric, such as AQI or PM2.5.
- The features are numeric pollutants and weather-related readings.
- Linear regression can be used as a simple baseline.
- Residual ranking can identify unusual observations that the model cannot explain well.
- Taichung and Changhua are directly meaningful as location filters, not only as a loose background story.

## Design boundary

The AQI case and the synthetic `n/a/b/var` sandbox are intentionally separated. The sandbox keeps the baseline regression workflow clear, while the AQI case gives the project a more practical operating context.
