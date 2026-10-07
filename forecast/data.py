"""
data.py — pull and clean the raw inputs.

Three sources:
  1. EIA-930 hourly demand for ISO-NE (type "D")           -> the target
  2. EIA-930 ISO-NE day-ahead demand forecast (type "DF")  -> the benchmark
  3. Open-Meteo weather, population-weighted across New England:
       - "forecast": archived day-ahead forecasts (Previous Runs API, 2024+)
       - "archive":  observed/reanalysis weather (any year; perfect-hindsight
                     upper bound, NOT a realistic forecast input)
       - live forecasts for the operational forecast script

All timestamps are returned as tz-aware UTC so joins never drift across DST.
"""

import time

import numpy as np
import pandas as pd
import requests

from . import config as C

EIA_REGION_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OM_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _get(url, params, retries=5):
    """
    GET with backoff. Retries on rate limits, server errors, and responses that
    aren't valid JSON (Open-Meteo sometimes returns an empty or plain-text body
    when it's throttling). If it still fails, show what the server actually said.
    """
    last_problem = ""
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code in (429, 500, 502, 503, 504):
            last_problem = f"HTTP {r.status_code}: {r.text[:300]}"
        elif r.status_code >= 400:
            raise RuntimeError(
                f"{url} returned HTTP {r.status_code}:\n{r.text[:500]}")
        else:
            try:
                return r.json()
            except ValueError:
                last_problem = (f"HTTP {r.status_code}, non-JSON body "
                                f"({len(r.text)} chars): {r.text[:300]!r}")
        if attempt < retries - 1:
            wait = 2 ** attempt * 10
            print(f"  retrying in {wait}s - {last_problem[:120]}")
            time.sleep(wait)
    raise RuntimeError(f"{url} failed after {retries} tries.\nLast response: {last_problem}")


# ---------------------------------------------------------------------------
# EIA
# ---------------------------------------------------------------------------
def pull_eia_series(series_type, start, end, api_key):
    """
    Hourly ISO-NE series from EIA-930. series_type: "D" (demand) or "DF"
    (ISO-NE's day-ahead demand forecast). start/end: "YYYY-MM-DD" (UTC dates).
    """
    records, offset = [], 0
    while True:
        params = {
            "api_key": api_key, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": "ISNE", "facets[type][]": series_type,
            "start": f"{start}T00", "end": f"{end}T23",
            "sort[0][column]": "period", "sort[0][direction]": "asc",
            "offset": offset, "length": 5000,
        }
        batch = _get(EIA_REGION_URL, params)["response"]["data"]
        records.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    # EIA hourly "period" is UTC and marks the END of the hour
    idx = pd.to_datetime(df["period"], format="%Y-%m-%dT%H").dt.tz_localize("UTC")
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index()


def clean_demand(s):
    """
    Remove reporting errors from the demand series. We set bad points to NaN
    rather than interpolating: the target should never contain made-up values,
    and the models handle NaN lag features natively.
    """
    s = s.copy()
    n0 = s.notna().sum()
    s[(s < C.DEMAND_MIN_MW) | (s > C.DEMAND_MAX_MW)] = np.nan
    med = s.rolling(5, center=True, min_periods=3).median()
    s[(s - med).abs() / med > C.SPIKE_TOLERANCE] = np.nan
    removed = n0 - s.notna().sum()
    if removed:
        print(f"  clean_demand: set {removed} implausible hourly values to NaN")
    # Reindex to a complete hourly grid so time shifts line up exactly
    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    return s.reindex(full)


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
def _weights():
    w = np.array([p[2] for p in C.WEATHER_POINTS.values()], dtype=float)
    return w / w.sum()


def _weighted_frame(payload, suffix=""):
    """
    Turn a multi-location Open-Meteo response into ONE population-weighted
    hourly frame. If a city is missing a value at some hour, the weights are
    renormalized over the cities that do have it.
    """
    if isinstance(payload, dict):
        payload = [payload]
    w = _weights()
    out = {}
    for om_var, col in C.WEATHER_VARS.items():
        key = om_var + suffix
        mat = np.array(
            [np.array(loc["hourly"][key], dtype=float) for loc in payload]
        )  # shape: (n_cities, n_hours)
        mask = ~np.isnan(mat)
        wsum = (w[:, None] * mask).sum(axis=0)
        val = np.nansum(w[:, None] * np.nan_to_num(mat), axis=0)
        out[col] = np.where(wsum > 0, val / np.where(wsum > 0, wsum, 1), np.nan)
    idx = pd.to_datetime(payload[0]["hourly"]["time"]).tz_localize("UTC")
    return pd.DataFrame(out, index=idx)


def _coord_params():
    pts = list(C.WEATHER_POINTS.values())
    return {
        "latitude": ",".join(str(p[0]) for p in pts),
        "longitude": ",".join(str(p[1]) for p in pts),
        "timezone": "UTC",
    }


def _date_chunks(start, end, months=3):
    """Split a long date range so each request stays a reasonable size."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        chunk_end = min(s + pd.DateOffset(months=months) - pd.Timedelta(days=1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + pd.Timedelta(days=1)


def pull_weather_history(start, end, mode="forecast"):
    """
    Historical weather for training/backtesting.

    mode="forecast": what the models PREDICTED one day ahead (realistic).
    mode="archive":  what actually HAPPENED (perfect-hindsight upper bound).
    """
    if mode == "forecast":
        if pd.Timestamp(start) < pd.Timestamp(C.PREVIOUS_RUNS_START):
            raise ValueError(
                f"Archived day-ahead weather forecasts start {C.PREVIOUS_RUNS_START}; "
                "use mode='archive' for earlier years (and label results accordingly)."
            )
        url, suffix = OM_PREVIOUS_RUNS_URL, "_previous_day1"
    elif mode == "archive":
        url, suffix = OM_ARCHIVE_URL, ""
    else:
        raise ValueError("mode must be 'forecast' or 'archive'")

    frames = []
    for s, e in _date_chunks(start, end):
        params = {
            **_coord_params(),
            "hourly": ",".join(v + suffix for v in C.WEATHER_VARS),
            "start_date": s, "end_date": e,
        }
        frames.append(_weighted_frame(_get(url, params), suffix))
        time.sleep(1)  # be polite to the free API
    wx = pd.concat(frames)
    return wx[~wx.index.duplicated(keep="last")].sort_index()


def pull_weather_live(past_days=8, forecast_days=3):
    """Current forecast run, for the operational (tomorrow) forecast."""
    params = {
        **_coord_params(),
        "hourly": ",".join(C.WEATHER_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
    }
    return _weighted_frame(_get(OM_FORECAST_URL, params))


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def build_hourly_frame(start, end, api_key, weather_mode="forecast"):
    """
    One hourly UTC-indexed frame: demand_mw, iso_forecast_mw, weather columns.

    EIA timestamps mark the END of an hour while Open-Meteo marks the START of
    an instantaneous reading; for hourly load modeling the one-hour difference
    is conventionally ignored, but we note it here so it's a deliberate choice.
    """
    print("Pulling ISO-NE demand (EIA type D)...")
    demand = clean_demand(pull_eia_series("D", start, end, api_key))
    print("Pulling ISO-NE day-ahead forecast (EIA type DF)...")
    iso_df = pull_eia_series("DF", start, end, api_key)
    print(f"Pulling population-weighted weather (mode={weather_mode})...")
    wx = pull_weather_history(start, end, mode=weather_mode)

    df = pd.DataFrame({C.TARGET: demand})
    df["iso_forecast_mw"] = iso_df.reindex(df.index)
    df = df.join(wx, how="left")
    df.index.name = "period_utc"
    return df
