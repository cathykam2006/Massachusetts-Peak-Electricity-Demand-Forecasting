# Day-Ahead Electricity Demand Forecasting for New England

An operational forecast of tomorrow's hourly electricity demand on the ISO New England grid, built only from free public data. Every morning it predicts all 24 hours of the next day with an 80% prediction range, recommends a plan-for peak with a risk-calibrated buffer, and scores itself against ISO-NE's own day-ahead forecast.

🔗 **Live app:** [Open the interactive app](https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/)

## Why it matters

Grid operators commit power plants a day in advance. Under-forecasting demand risks emergency purchases and reliability problems; over-forecasting wastes capacity and raises costs for ratepayers. Peak hours matter most, because they drive capacity costs that end up on customers' bills, and a shortfall at the peak is far more costly than holding spare capacity.

## Key results

A rolling-origin backtest replays October 2025 to October 2026 exactly as the forecast would have run live: retrain every 28 days, forecast each next day out of sample, step forward.

| Forecast | Hourly MAPE | MAE (MW) | Daily peak MAPE | Peak hour within ±1h |
|---|---|---|---|---|
| Naive: same hour last week | 11.05% | 1,482 | 8.36% | 91.2% |
| **This model** | **3.17%** | **415** | **2.04%** | **96.4%** |
| ISO-NE's published day-ahead forecast | 3.04% | 373 | 1.24% | 97.8% |

- **71% less error than the naive baseline**, and within 0.14 percentage points of the grid operator on hourly error, using only public data. ISO-NE still leads on daily peaks (1.24% vs. 2.04%), where its proprietary inputs, such as rooftop-solar estimates and real-time load, matter most.
- **Calibrated uncertainty:** the 80% prediction range contained actual demand in 79.2% of hours.
- **Unbiased:** average error of +6 MW.
- **A dedicated daily-peak model**, averaged with the hourly model's peak, cut daily-peak error from 2.17% to 2.04% and passed the same two-year test.
- **Blending four weather models** (GFS, ECMWF, ICON, GEM) cut hourly error from 3.60% to 3.18% and passed a pre-registered two-year test (see below).
- **Solar matters:** adding sunshine forecasts cut error in spring, when rooftop solar is strong and loads are moderate, from 4.76% to about 4%. Rooftop solar isn't metered by ISO-NE, so sunny middays show up as lower demand.

Accuracy by season:

| Season | Hourly MAPE | Daily peak MAPE |
|---|---|---|
| Winter | 2.93% | 2.01% |
| Spring | 3.51% | 1.86% |
| Summer | 3.34% | 3.03% |
| Fall | 2.93% | 1.78% |

Summer peaks remain the hardest to predict, since heat-wave demand depends on humidity and multi-day heat buildup.

**A note on the ISO-NE comparison.** ISO-NE's forecast runs 5–8% below EIA's reported demand overnight, in every month of the year. That pattern points to a definitional difference between the two series (most likely overnight pumping at pumped-storage hydro plants, counted in EIA's demand but not in ISO's load forecast) rather than poor forecasting, and it explains most of ISO's −293 MW average bias. The fairest comparison is the evening peak hours, where ISO-NE's error is about 1.3%.

## Planning buffer

Because running short of power costs far more than holding extra capacity, the app reports a **plan-for peak** above the expected peak. The buffer starts from the day's highest 90th-percentile hour and adds a margin learned from the previous 90 days of actual peaks, sized so the actual peak goes above the plan at a chosen rate. The margin adapts as it goes (adaptive conformal prediction): it tightens after a miss and relaxes slightly after covered days, which keeps it on target when misses come in streaks during heat waves and cold snaps. Each day uses only days already observed at forecast time.

The app's Accuracy page shows the current miss rate and average buffer at each risk level (1 day in 10, 20, or 50), recomputed whenever the model is retrained.

Because the plan-for level sits above the whole day's peak, it covers every hour of the day. The remaining misses cluster on the most extreme days, where weather forecasts themselves are least reliable; this is the reason grid operators hold operating reserves on top of any forecast.

## Guarding against overfitting

An audit of the first version of this model found that it fit its training data to about 1% error but new days to about 4%, a gap of roughly 3 percentage points. The current model was chosen with a protocol designed not to repeat that:

1. Candidate settings were **fixed in advance** from standard practice, not tuned on the data.
2. Each candidate was **judged on a year not used to choose it** (Oct 2024 – Oct 2025), then **confirmed on the following year** (Oct 2025 – Oct 2026).
3. A change was adopted only if it improved results **consistently across months**, not just on average.

| | Previous model, earlier year | New model, earlier year | Previous model, recent year | New model, recent year |
|---|---|---|---|---|
| Hourly MAPE | 4.36% | **4.16%** | 3.79% | **3.60%** |
| Months where the new model wins | | **11 of 13** | | **10 of 14** |
| Shortfall on top-10% demand days | 862 MW | **550 MW** | 546 MW | **250 MW** |
| Train-test error gap | 3.62 pts | **2.38 pts** | 2.79 pts | **1.49 pts** |

A more flexible alternative (linear trees without regularization) looked better on the recent year but won only 6 of 13 months on the earlier one and widened the train-test gap, so it was rejected.

Later candidates went through the same test, with the rule fixed before any results: lower error in **both** years and better in **most months** of each.

| Candidate | Earlier year | Recent year | Months better | Decision |
|---|---|---|---|---|
| Blend of four weather models (mean + spread) | 4.18% → 3.82% | 3.60% → 3.22% | 12 of 14, 12 of 14 | **Adopted** |
| Daily-peak model, averaged with the hourly peak | 2.69% → 2.51% (peak) | 2.14% → 2.03% (peak) | 11 of 14, 10 of 13 | **Adopted** |
| Rooftop-solar growth feature | 4.16% → 4.23% | 3.60% → 3.70% | 6 of 13, 5 of 14 | Rejected |
| Bias correction by time of day | 4.16% → 4.12% | 3.60% → 3.61% | 7 of 13, 5 of 14 | Rejected |

The weather test can be rerun at any time with `python compare_weather.py`.

## How it works

**Forecast setup.** At about 10 AM ET each day, the model forecasts all 24 hours of the next day. It uses only information that exists at that moment:

- calendar facts: hour, weekday, holidays, season
- the **weather forecast** for tomorrow (temperature, dew point, humidity, wind, cloud cover and sunshine) from four global weather models, averaged across eight New England cities weighted by population, plus how much the models disagree
- demand through two days ago, plus the current morning's demand so far

**Data sources.**

| Data | Source | Notes |
|---|---|---|
| Hourly ISO-NE demand | [EIA-930 API](https://www.eia.gov/opendata/) (series `D`) | Target variable |
| ISO-NE day-ahead forecast | [EIA-930 API](https://www.eia.gov/opendata/) (series `DF`) | Benchmark |
| Archived weather forecasts | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | What GFS, ECMWF, ICON and GEM predicted the day before (2024 onward) |
| Live weather forecasts | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | For the daily forecast |

Training on archived *forecasts* rather than observed weather is deliberate: the model learns with the same weather-forecast error it faces in live use.

**Weather blend.** Each of the four weather models is pulled separately and averaged. Their spread on temperature, cloud cover and sunshine is a feature, so the model trusts the weather less when forecasters disagree. If one weather model is unavailable, the forecast continues with the others.

**Peak model.** A second model predicts each day's peak from that day's 6 PM conditions. Its estimate is averaged with the hourly model's peak, and the hourly curve is shifted toward that average around the peak hour.

**Model.** A single regularized LightGBM model with linear trees. Regularization (15 leaves, at least 100 hours per leaf, an L2 penalty) limits overfitting. Linear trees fit a small linear trend in each leaf instead of a constant, so record cold snaps and heat waves aren't flattened at the edge of the training data. LightGBM quantile models provide the 80% range.

**Calibration.** Using only days already observed at forecast time, the forecast is corrected for recent bias, and the 80% range is resized with conformalized quantile regression so that it really covers 80% of hours.

**Explainability.** SHAP values show which inputs drive the forecast, how demand responds to temperature, and when extreme peaks occur.

**Supply side.** The app also shows where the power comes from each hour (nuclear, hydro, solar, natural gas, wind, net imports), from EIA's hourly generation-by-source data.

**Operations.** A GitHub Action issues tomorrow's forecast every morning and commits it before the actuals exist, building a live, publicly scored track record. A second Action retrains the model monthly.

## Version 1 and what changed

The first version of this project reported a 1.27% MAPE. A review found that it used information that would not exist at forecast time: the previous hour's actual demand, a 24-hour rolling average that included the hour being predicted, and observed (not forecast) weather and renewable generation. That made it a one-hour-ahead nowcast rather than a forecast.

Version 2 rebuilds the project as a true day-ahead forecast. The honest error is higher, but it reflects what the model can actually do, and it allows a fair comparison with ISO-NE. The original app is kept as `app_v1.py`, along with the version 1 notebooks.

## Repository structure

```
forecast/
  config.py          settings: forecast timing, weather locations, backtest options
  data.py            EIA and Open-Meteo data pulls and cleaning
  features.py        day-ahead features with no look-ahead
  models.py          regularized linear-tree LightGBM and quantile models
  calibration.py     online bias correction and conformal intervals
  evaluate.py        rolling-origin backtest and metrics
  explain.py         SHAP explanations (companion model)
  supply.py          generation by source and net imports
train_dayahead.py    build dataset, backtest, train final models
compare_weather.py   pre-registered test: single vs. blended weather, two years
update_supply.py     refresh generation-by-source data for the supply page
forecast_tomorrow.py issue tomorrow's forecast (run daily by GitHub Actions)
app.py               Streamlit app, including the planning buffer
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
python train_dayahead.py --weather-blend   # pull data, backtest, train (about 20 minutes)
python forecast_tomorrow.py     # issue tomorrow's forecast
streamlit run app.py            # view the app locally
```

To run the daily forecast automatically on GitHub, add `EIA_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) and allow Actions to write to the repository (Settings → Actions → General → Workflow permissions → Read and write).

## Limitations

- Archived day-ahead weather forecasts are available only from January 2024, which limits the training history to under three years and the planning-buffer backtest to about one year of extreme days.
- Weather comes from four public models at eight cities. ISO-NE uses many more stations, its own meteorologists, and its own forecasts of behind-the-meter solar.
- Demand data comes from EIA-930, which can be revised after first publication.
- Buffer misses still cluster on the most extreme days; the buffer complements, and does not replace, grid operating reserves.

Data: U.S. Energy Information Administration; weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
