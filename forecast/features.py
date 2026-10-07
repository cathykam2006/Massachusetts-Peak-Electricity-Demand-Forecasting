"""
features.py — day-ahead features with no look-ahead.

Rule: a row predicts hour h of day D. Its features may only use information
available at the forecast origin, 10 AM on D-1:

  * calendar facts about D                         (always known)
  * the WEATHER FORECAST for D                     (from the forecast archive)
  * demand from D-2 and earlier                    (fully observed)
  * demand from D-1 before D1_CUTOFF_HOUR          (the morning so far)

What the old pipeline did that this one deliberately does NOT:
  * load_lag_1h        - the previous hour's actual demand (unknown a day ahead)
  * load_roll_mean_24h - rolling(24) without a shift, which included the
                         target hour itself (direct target leakage)
  * same-hour observed weather and solar/wind generation (unknown in advance)
"""

import holidays
import numpy as np
import pandas as pd

from . import config as C

WEATHER_COLS = list(C.WEATHER_VARS.values())


def _date_key(local_index):
    """Local calendar date as a tz-naive Timestamp (handy for groupby/map)."""
    return pd.DatetimeIndex(local_index.date)


def build_features(hourly):
    """
    hourly: UTC-indexed frame with C.TARGET and (some of) WEATHER_COLS,
            on a complete hourly grid (see data.clean_demand).
    Returns a frame indexed like `hourly` with feature columns added, plus
    `ts_local` and `target_date` helper columns.
    """
    df = hourly.copy()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    local = df.index.tz_convert(C.TZ)
    date = _date_key(local)
    df["ts_local"] = local
    df["target_date"] = date

    # --- Calendar ---------------------------------------------------------
    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    doy = local.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    years = range(local.year.min() - 1, local.year.max() + 2)
    hol = holidays.US(years=years)
    df["is_holiday"] = pd.Series(date, index=df.index).map(lambda d: d in hol).astype(int)
    # Load sags through the Christmas-New Year stretch even on working days
    md = local.month * 100 + local.day
    df["is_holiday_season"] = ((md >= 1224) | (md <= 101)).astype(int)

    # --- Weather forecast for the target hour and day -------------------
    have_wx = [c for c in WEATHER_COLS if c in df.columns]
    if "temp_c" in have_wx:
        t = df["temp_c"]
        df["hdh"] = (C.DEGREE_BASE_C - t).clip(lower=0)   # heating degree hours
        df["cdh"] = (t - C.DEGREE_BASE_C).clip(lower=0)   # cooling degree hours
        daily = t.groupby(date).agg(["max", "min", "mean"])
        daily.columns = ["temp_max_d", "temp_min_d", "temp_mean_d"]
        for col in daily.columns:
            df[col] = date.map(daily[col]).values
        # Buildings carry heat/cold between days: a 3-day weighted mean
        # temperature captures heat-wave and cold-snap buildup.
        m = daily["temp_mean_d"]
        heat_index = 0.6 * m + 0.3 * m.shift(1, freq="D") + 0.1 * m.shift(2, freq="D")
        df["temp_mean_3d_wtd"] = date.map(heat_index).values
        df["cdd_d"] = (df["temp_mean_d"] - C.DEGREE_BASE_C).clip(lower=0)
        df["hdd_d"] = (C.DEGREE_BASE_C - df["temp_mean_d"]).clip(lower=0)

    # --- Demand history (only what's observed by the forecast origin) ----
    y = df[C.TARGET]
    # Positional shifts are exact time shifts because the grid is complete.
    df["load_same_hour_d2"] = y.shift(48)
    df["load_same_hour_d7"] = y.shift(168)
    df["load_same_hour_d14"] = y.shift(336)

    by_day = y.groupby(date)
    d_stats = pd.DataFrame({"peak": by_day.max(), "mean": by_day.mean()})
    df["load_peak_d2"] = (date - pd.Timedelta(days=2)).map(d_stats["peak"]).values
    df["load_mean_d2"] = (date - pd.Timedelta(days=2)).map(d_stats["mean"]).values
    df["load_peak_d7"] = (date - pd.Timedelta(days=7)).map(d_stats["peak"]).values

    # D-1 early-morning demand: the freshest data an operator has at 10 AM
    morning = y[local.hour < C.D1_CUTOFF_HOUR].groupby(date[local.hour < C.D1_CUTOFF_HOUR])
    last_obs = y[local.hour == C.D1_CUTOFF_HOUR - 1].groupby(
        date[local.hour == C.D1_CUTOFF_HOUR - 1]).last()
    d1 = date - pd.Timedelta(days=1)
    df["load_d1_morning_mean"] = d1.map(morning.mean()).values
    df["load_d1_last_obs"] = d1.map(last_obs).values
    # Same morning a week earlier, so the model can read "this morning vs normal"
    df["load_d1_morning_vs_lastweek"] = (
        df["load_d1_morning_mean"]
        / (date - pd.Timedelta(days=8)).map(morning.mean()).values
    )
    return df


def feature_columns(df):
    """The model inputs that exist in this frame, in a stable order."""
    candidates = [
        "hour", "dayofweek", "month", "doy_sin", "doy_cos",
        "is_weekend", "is_holiday", "is_holiday_season",
        *WEATHER_COLS,
        "hdh", "cdh", "temp_max_d", "temp_min_d", "temp_mean_d",
        "temp_mean_3d_wtd", "cdd_d", "hdd_d",
        "load_same_hour_d2", "load_same_hour_d7", "load_same_hour_d14",
        "load_peak_d2", "load_mean_d2", "load_peak_d7",
        "load_d1_morning_mean", "load_d1_last_obs", "load_d1_morning_vs_lastweek",
    ]
    return [c for c in candidates if c in df.columns]


def modeling_rows(df):
    """Rows usable for training/evaluation: target observed."""
    return df[df[C.TARGET].notna()]
