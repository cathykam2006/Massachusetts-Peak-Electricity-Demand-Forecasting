"""
forecast_tomorrow.py — issue tomorrow's 24-hour forecast, like an operator would.

Run each morning (a GitHub Action does this automatically at ~10:30 AM ET):
    export EIA_API_KEY="your_key_here"
    python forecast_tomorrow.py

What it does:
  1. Pulls the last three weeks of ISO-NE demand and the current weather forecast.
  2. Builds the same features as training, for tomorrow's 24 hours.
  3. Predicts, then calibrates (bias + interval) using the live forecast log.
  4. Writes data/dayahead/latest_forecast.csv and appends to forecast_log.csv.

The log is the project's real out-of-sample track record: every forecast is
saved BEFORE the actuals exist, and actuals are filled in on later runs.
"""

import os

import joblib
import numpy as np
import pandas as pd

from forecast import config as C
from forecast.calibration import apply_calibration, calibration_state
from forecast.data import clean_demand, pull_eia_series, pull_weather_live
from forecast.features import build_features
from forecast.models import predict

LOG_PATH = f"{C.DATA_DIR}/forecast_log.csv"
LATEST_PATH = f"{C.DATA_DIR}/latest_forecast.csv"


def main():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set the EIA_API_KEY environment variable first.")
    bundle = joblib.load(f"{C.MODEL_DIR}/bundle.pkl")

    now_local = pd.Timestamp.now(tz=C.TZ)
    tomorrow = (now_local + pd.Timedelta(days=1)).normalize().tz_localize(None)
    start = (now_local - pd.Timedelta(days=22)).strftime("%Y-%m-%d")
    end = (now_local + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"Issuing forecast for {tomorrow.date()} at {now_local:%Y-%m-%d %H:%M %Z}")

    # 1. Inputs -----------------------------------------------------------
    demand = clean_demand(pull_eia_series("D", start, end, api_key))
    iso_df = pull_eia_series("DF", start, end, api_key)
    wx = pull_weather_live(past_days=4, forecast_days=3)
    print(f"  latest demand observation: "
          f"{demand.last_valid_index().tz_convert(C.TZ):%Y-%m-%d %H:%M}")

    # Hourly grid running through the end of tomorrow (local)
    grid_end = (tomorrow + pd.Timedelta(days=1, hours=1)).tz_localize(C.TZ).tz_convert("UTC")
    grid = pd.date_range(demand.index.min(), grid_end, freq="h", tz="UTC")
    hourly = pd.DataFrame({C.TARGET: demand.reindex(grid)}, index=grid)
    hourly["iso_forecast_mw"] = iso_df.reindex(grid)
    hourly = hourly.join(wx, how="left")

    # 2. Features + raw prediction ---------------------------------------
    feat = build_features(hourly)
    tom = feat[feat["target_date"] == tomorrow].copy()
    missing = tom[bundle["feature_cols"]].isna().mean()
    if (missing > 0).any():
        print("  note — features with missing values (model handles NaN):\n"
              + missing[missing > 0].round(2).to_string())
    raw = predict(bundle["models"], tom[bundle["feature_cols"]])
    raw = raw.rename(columns={"pred": "pred_raw", "pred_lo": "pred_lo_raw",
                              "pred_hi": "pred_hi_raw"})

    # 3. Calibration from the live log (fallback: backtest) --------------
    log = pd.read_csv(LOG_PATH, index_col="period_utc", parse_dates=["period_utc"]) \
        if os.path.exists(LOG_PATH) else pd.DataFrame()
    if not log.empty:
        log.index = pd.to_datetime(log.index, utc=True)
        log["target_date"] = pd.to_datetime(log["target_date"])
        log[C.TARGET] = demand.reindex(log.index).combine_first(log.get(C.TARGET))
        bias, q = calibration_state(log, tomorrow)
    else:
        bias, q = 0.0, 0.0
    source = "live log"
    if (bias, q) == (0.0, 0.0):
        fb = bundle["fallback_calibration"]
        bias, q, source = fb["bias"], fb["q"], "backtest fallback"
    print(f"  calibration ({source}): bias {bias:+.0f} MW, interval adj {q:+.0f} MW")

    out = apply_calibration(
        tom[["ts_local", "target_date", "iso_forecast_mw"]].join(raw), bias, q)
    out["issued_at"] = now_local.isoformat(timespec="minutes")
    out["models_trained_through"] = bundle["trained_through"]
    out[C.TARGET] = np.nan
    out.index.name = "period_utc"

    # 4. Save -------------------------------------------------------------
    os.makedirs(C.DATA_DIR, exist_ok=True)
    out.to_csv(LATEST_PATH)
    if not log.empty:
        log = log[~log.index.isin(out.index)]   # a re-run replaces that day's forecast
        out = pd.concat([log, out]).sort_index()
    out.to_csv(LOG_PATH)

    peak = out.loc[out["target_date"] == tomorrow]
    p = peak.loc[peak["pred"].idxmax()]
    print(f"\nTomorrow's peak: {p['pred']:,.0f} MW "
          f"({p['pred_lo']:,.0f}-{p['pred_hi']:,.0f}) at {p['ts_local']:%H:%M}")
    if peak["iso_forecast_mw"].notna().sum() >= 20:
        print(f"ISO-NE's day-ahead peak forecast: {peak['iso_forecast_mw'].max():,.0f} MW")
    else:
        print("ISO-NE's day-ahead forecast for tomorrow isn't fully posted yet.")


if __name__ == "__main__":
    main()
