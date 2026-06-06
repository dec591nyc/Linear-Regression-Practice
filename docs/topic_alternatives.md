# Public Data Topic Alternatives

## Selection Criteria

- Prefer daily-life topics.
- Prefer Taichung, Changhua, Nantou, or central Taiwan.
- Prefer public data that can support numeric modeling.
- Prefer row counts between 20,000 and 100,000.
- Avoid tiny demo files.

## Candidate Topics

| Topic | Possible prediction target | Why it is useful | Data concern |
|---|---|---|---|
| Taichung YouBike station demand | station rental volume, available bikes, or empty slots | Very close to daily life and easy to explain as commuting demand | Real historical trip-level data may not be fully open; station and monthly summary data are easier to obtain |
| Taichung parking availability | remaining parking spaces or occupancy rate | Practical for drivers and shopping districts | Many open feeds are real-time snapshots; historical collection may be needed |
| Taichung 1999 citizen complaints | monthly case count or processing volume | Connects to public-service demand and city operations | Often aggregated monthly, so row count may be too small unless broken down by agency/district/category |
| Taichung housing market quarterly statistics | average unit price or transaction volume | Familiar life topic and directly numeric | Usually quarterly and district-level; row count may be low |
| Central Taiwan AQI | AQI or PM2.5 | Still a strong fit because pollutant fields are numeric and regression-friendly | Current local CSV is Taichung-only historical data; ideal recent central dataset still needs a larger official extract |
| Tourism lodging or attractions | room count, site density, or visitor-related proxy | Lifestyle topic, can connect to travel/business | Many tourism datasets are cross-sectional rather than repeated observations |

## Current Best Options

1. Keep AQI with `CentraArea_Data.csv` as a pragmatic compromise. It has 37,286 rows and supports regression, but it is Taichung-only historical data.
2. Switch to Taichung YouBike demand if historical station usage can be obtained at sufficient granularity.
3. Switch to parking availability only if a historical snapshot dataset can be built or found.

## Source Leads

- Taichung public bicycle rental summary: https://data.nat.gov.tw/dataset/6561692
- Taichung YouBike station and real-time availability: https://odportal.tw/dataset/IePhphdK
- Taichung open data portal: https://opendata.taichung.gov.tw/
- Taichung parking availability: https://scidm.nchc.org.tw/dataset/best_wish83931
- Taichung 1999 citizen complaint statistics: https://odportal.tw/dataset/NovbICap
- Tourism hotel and homestay dataset: https://data.gov.tw/dataset/7780
- Taichung real-estate quarterly dynamic analysis: https://data.gov.tw/dataset/83768
