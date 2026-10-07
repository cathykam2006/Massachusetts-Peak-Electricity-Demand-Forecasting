# Predicting Peak Electricity Demand in Massachusetts

Using weather, load, renewable generation, and consumer behavior data to forecast hourly electricity demand for ISO New England, with a focus on identifying the conditions that drive extreme peak demand events.

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1qbmBJ6c9Wma3i2UhiRgKJda6EqOqR0HhGvHgHtlM0N7SlizAsOUzRUg&s=10" alt="Massachusetts Grid" width="500">
</p>

## Overview

Electricity grid operators must balance supply and demand in real time, every hour of every day. Under-forecasting demand risks blackouts and emergency power purchases; over-forecasting wastes generation capacity and drives up costs for ratepayers. This project builds and compares four forecasting approaches — from a naive seasonal baseline to a gradient-boosted ensemble — to predict hourly electricity demand in the ISO-NE region, and then uses model explainability (SHAP) to answer a more specific question: **what conditions are most associated with extreme peak demand, and does that answer change by season?**

🔗 **Live app:** [Open the interactive app](https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/)

## Key Findings

- An **ensemble of XGBoost and LightGBM** achieved the best forecasting accuracy (MAPE: 1.27%), roughly an 8x improvement over a naive seasonal-average baseline (MAPE: 10.47%).
- **SARIMAX**, a classical statistical time-series model, underperformed even the naive baseline at both a 90-day and a 24-hour forecast horizon — a known limitation of ARIMA-family models at scale without access to recent ground-truth values.
- **Extreme peak demand** clusters around 7 PM on weekdays in both winter and summer, but for opposite physical reasons: winter peaks are driven by heating load on cold evenings, while summer peaks are driven by cooling load on hot evenings. Roughly 95% of extreme-demand hours in both seasons occur on weekdays.

## Version 2: Operational Day-Ahead Forecast

Version 1 used features that would not exist at forecast time: the previous hour's actual demand, a 24-hour rolling mean that included the target hour itself, and observed (not forecast) weather and generation. That made it a one-hour-ahead nowcast, which is why its MAPE was 1.27%. Version 2 rebuilds the project as a forecast an operator could actually issue.

**Forecast setup.** Each morning at about 10 AM ET, the model forecasts all 24 hours of the next day. It uses only information available at that moment: calendar facts, the *weather forecast* for tomorrow, demand through two days ago, and the early-morning demand already observed today.

**Inputs.**
- ISO-NE hourly demand from EIA-930.
- Archived day-ahead weather forecasts from the Open-Meteo Previous Runs API (2024 onward). These record what the weather models predicted the day before, not what actually happened, so the model is trained and tested with realistic weather-forecast error.
- Population-weighted weather across eight New England load centers, rather than a single Boston point.

**Evaluation.**
- A rolling-origin backtest over the most recent year: refit every 28 days, forecast each day out-of-sample, and step forward.
- Benchmarks are a seasonal-naive forecast and **ISO-NE's own published day-ahead forecast** (EIA series `DF`).
- Metrics are hourly MAPE, daily-peak MAPE, peak-hour accuracy, and prediction-interval coverage.

**Calibration.** An online step corrects each forecast for the model's recent bias, which comes from load drifting down as behind-the-meter solar grows. It also calibrates an 80% prediction interval using conformalized quantile regression. Both use only errors from days fully observed at forecast time.

**Operations.** A GitHub Action issues tomorrow's forecast every morning and commits it to `data/dayahead/`. Every forecast is logged *before* its actuals exist, building a live out-of-sample track record. A second action retrains monthly.

```
forecast/            config, data pulls, features, models, calibration, evaluation
train_dayahead.py    build dataset -> backtest -> train final models
forecast_tomorrow.py issue tomorrow's forecast (run daily by GitHub Actions)
```

```bash
export EIA_API_KEY="your_key"
python train_dayahead.py            # backtest + final models (2024 -> yesterday)
python train_dayahead.py --weather archive --start 2021-01-01   # perfect-weather upper bound
python forecast_tomorrow.py         # tomorrow's forecast
```

## Data Sources (Version 1)

This project uses three real, publicly available data sources:

| Data | Source | Frequency |
|---|---|---|
| Electricity demand (ISO-NE) | [EIA Hourly Electric Grid Monitor API](https://www.eia.gov/electricity/gridmonitor/) ([docs](https://www.eia.gov/opendata/)) | Hourly |
| Weather (Boston, MA) | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Hourly |
| Renewable generation (solar/wind, ISO-NE) | [EIA Electricity Data API](https://www.eia.gov/opendata/browser/electricity/rto/fuel-type-data) | Hourly |

Consumer behavior is represented through calendar-based proxy features (hour of day, day of week, weekend/holiday flags), since individual smart-meter-level data is not publicly available.

> **Note:** An earlier version of this project used a Kaggle dataset that was found, through a structured validation process, to be synthetically generated with no real temporal or spatial structure. It was replaced with the real data sources above. This investigation is documented in the notebook itself as part of the project's data-quality process.

## Methodology

1. **Data sourcing & validation** — pulling and verifying real hourly demand, weather, and generation data via free public APIs.
2. **Feature engineering** — lag features (1h, 24h, 168h), rolling averages, heating/cooling degree days, and calendar/behavioral flags.
3. **Model comparison** — baseline (seasonal historical average) → SARIMAX → XGBoost / LightGBM → ensemble.
4. **Explainability** — SHAP analysis on the top 5% highest-demand hours, run separately on winter and summer test windows to characterize seasonal drivers of extreme demand.

## Results

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Baseline (seasonal avg) | 1217.0 | 1468.5 | 10.47% |
| SARIMAX (90-day horizon) | 5035.6 | 5381.4 | 43.61% |
| SARIMAX (24-hour horizon) | 1977.3 | 2167.4 | 18.35% |
| XGBoost | 159.2 | 207.3 | 1.31% |
| LightGBM | 157.1 | 204.0 | 1.29% |
| **Ensemble** | **155.0** | **201.4** | **1.27%** |

## Repository Structure

```
.
├── notebook.ipynb          # Main analysis notebook (data pull, EDA, modeling, SHAP)
├── README.md                # This file
└── isone_full_dataset.csv   # Cached combined dataset (demand + weather + generation + calendar features)
```

*(Adjust the above to match your actual file/folder names.)*

## Requirements

```
pandas
numpy
matplotlib
seaborn
requests
xgboost
lightgbm
statsmodels
shap
holidays
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn requests xgboost lightgbm statsmodels shap holidays scikit-learn
```

## Getting an API Key

This project pulls live data from the EIA API, which requires a free API key:

1. Register at [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php) (instant, email only).
2. Paste your key into the `API_KEY` variable near the top of the notebook.

Open-Meteo's historical weather API requires no key.

## Limitations

- Weather data is sourced for Boston specifically, used as a proxy for the broader Massachusetts/ISO-NE service area.
- SARIMAX tuning was constrained by compute time; a more exhaustively tuned model may perform better than reported here.
- The extreme-peak explainability analysis covers two 2-month seasonal windows (fall/winter and summer) rather than a full year of rolling analysis.
