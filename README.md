# Day-Ahead Electricity Demand Forecasting for New England

An operational forecast of tomorrow's hourly electricity demand on the ISO New England grid, built only from free public data. Every morning it predicts all 24 hours of the next day, with an 80% prediction range, and scores itself against ISO-NE's own day-ahead forecast.

🔗 **Live app:** [Open the interactive app](https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/)

## Why it matters

Grid operators commit power plants a day in advance. Under-forecasting demand risks emergency purchases and reliability problems; over-forecasting wastes capacity and raises costs for ratepayers. Peak hours matter most, because they drive capacity costs that end up on customers' bills.

## Key results

A rolling-origin backtest replays October 2025 to October 2026 exactly as the forecast would have run live: retrain every 28 days, forecast each next day out of sample, step forward.

| Forecast | Hourly MAPE | MAE (MW) | Daily peak MAPE | Peak hour within ±1h |
|---|---|---|---|---|
| Naive: same hour last week | 10.99% | 1,476 | 8.32% | 91.0% |
| **This model** (calibrated LightGBM + XGBoost ensemble) | **3.79%** | **499** | **2.56%** | **95.6%** |
| ISO-NE's published day-ahead forecast | 3.04% | 374 | 1.25% | 97.5% |

- **65% less error than the naive baseline**, and within 0.75 percentage points of the grid operator, which uses far richer proprietary data.
- **Calibrated uncertainty:** the 80% prediction range contained actual demand in 79.1% of hours.
- **Unbiased:** average error of −2 MW, versus about −290 MW for ISO-NE's forecast over the same period.
- **Solar matters:** adding sunshine forecasts cut overall error from 4.08% to 3.79%, and spring error from 4.76% to 4.08%. Rooftop solar isn't metered by ISO-NE, so sunny middays show up as lower demand.

Accuracy by season (calibrated ensemble):

| Season | Hourly MAPE | Daily peak MAPE |
|---|---|---|
| Winter | 3.39% | 2.30% |
| Spring | 4.08% | 1.81% |
| Summer | 4.18% | 3.93% |
| Fall | 3.52% | 2.17% |

Summer peaks remain the hardest to predict, since heat-wave demand depends on humidity and multi-day heat buildup.

## How it works

**Forecast setup.** At about 10 AM ET each day, the model forecasts all 24 hours of the next day. It uses only information that exists at that moment:

- calendar facts: hour, weekday, holidays, season
- the **weather forecast** for tomorrow (temperature, dew point, humidity, wind, cloud cover and sunshine), averaged across eight New England cities weighted by population
- demand through two days ago, plus the current morning's demand so far

**Data sources.**

| Data | Source | Notes |
|---|---|---|
| Hourly ISO-NE demand | [EIA-930 API](https://www.eia.gov/opendata/) (series `D`) | Target variable |
| ISO-NE day-ahead forecast | [EIA-930 API](https://www.eia.gov/opendata/) (series `DF`) | Benchmark |
| Archived weather forecasts | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | What the weather models predicted the day before (2024 onward) |
| Live weather forecasts | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | For the daily forecast |

Training on archived *forecasts* rather than observed weather is deliberate: the model learns with the same weather-forecast error it faces in live use.

**Models.** The point forecast averages LightGBM and XGBoost. LightGBM quantile models provide the 80% range.

**Calibration.** Using only days already observed at forecast time, the forecast is corrected for recent bias, and the range is resized with conformalized quantile regression so that it really covers 80% of hours.

**Explainability.** SHAP values show which inputs drive the forecast, how demand responds to temperature, and when extreme peaks occur.

**Operations.** A GitHub Action issues tomorrow's forecast every morning and commits it before the actuals exist, building a live, publicly scored track record. A second Action retrains the models monthly.

## Version 1 and what changed

The first version of this project reported a 1.27% MAPE. A review found that it used information that would not exist at forecast time: the previous hour's actual demand, a 24-hour rolling average that included the hour being predicted, and observed (not forecast) weather and renewable generation. That made it a one-hour-ahead nowcast rather than a forecast.

Version 2 rebuilds the project as a true day-ahead forecast. The honest error is higher (3.79%), but it reflects what the model can actually do, and it allows a fair comparison with ISO-NE. The original app is kept as `app_v1.py`, along with the version 1 notebooks.

## Repository structure

```
forecast/
  config.py          settings: forecast timing, weather locations, backtest options
  data.py            EIA and Open-Meteo data pulls and cleaning
  features.py        day-ahead features with no look-ahead
  models.py          LightGBM/XGBoost ensemble and quantile models
  calibration.py     online bias correction and conformal intervals
  evaluate.py        rolling-origin backtest and metrics
train_dayahead.py    build dataset, backtest, train final models
forecast_tomorrow.py issue tomorrow's forecast (run daily by GitHub Actions)
app.py               Streamlit app
app_v1.py            original version 1 app, kept for reference
data/dayahead/       backtest results, metrics, live forecast log
models/dayahead/     trained models
.github/workflows/   daily forecast and monthly retrain
*.ipynb              version 1 notebooks (data cleaning, EDA, modeling)
train_and_save.py    version 1 training script
graphs/, *.pdf       version 1 charts and presentation decks
```

## Running it yourself

You need a free EIA API key from [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php). Open-Meteo needs no key.

```bash
pip install -r requirements.txt
export EIA_API_KEY="your_key"
python train_dayahead.py        # pull data, backtest, train (about 20 minutes)
python forecast_tomorrow.py     # issue tomorrow's forecast
streamlit run app.py            # view the app locally
```

To run the daily forecast automatically on GitHub, add `EIA_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) and allow Actions to write to the repository (Settings → Actions → General → Workflow permissions → Read and write).

## Limitations

- Archived day-ahead weather forecasts are available only from January 2024, which limits the training history to about two and three-quarter years.
- Weather is averaged over eight cities. ISO-NE uses many more stations plus its own forecasts of behind-the-meter solar.
- Demand data comes from EIA-930, which can be revised after first publication.
- ISO-NE's forecast may be defined slightly differently from EIA's demand series, which could explain part of its average bias.

Data: U.S. Energy Information Administration; weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
