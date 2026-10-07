"""
config.py — every assumption the day-ahead pipeline makes, in one place.

The forecasting setup mirrors how a real operator works: each morning at the
forecast "origin" (10 AM local, D-1), we issue a forecast for all 24 hours of
the next day (D). Every feature must be knowable at that moment.
"""

TZ = "America/New_York"
TARGET = "demand_mw"

# --- Forecast timing -------------------------------------------------------
# Demand observations are assumed available up to (but not including) this
# local hour on D-1. EIA-930 usually posts hourly data within a few hours, so 7
# (i.e. midnight-6 AM on D-1) leaves a comfortable publication buffer before a
# 10 AM issue time. If the data hasn't posted, LightGBM/XGBoost treat the
# missing features as NaN and still produce a forecast.
D1_CUTOFF_HOUR = 7

# --- Regional weather ------------------------------------------------------
# ISO-NE serves all six New England states, so a single Boston weather point
# under-represents the load. These are major load centers weighted by
# approximate metro-area population (rounded, in millions). Weights only need
# to be roughly right — they are normalized in code.
WEATHER_POINTS = {
    "Boston":      (42.3601, -71.0589, 4.9),
    "Providence":  (41.8240, -71.4128, 1.7),
    "Hartford":    (41.7658, -72.6734, 1.2),
    "Worcester":   (42.2626, -71.8023, 1.0),
    "New Haven":   (41.3083, -72.9279, 0.9),
    "Springfield": (42.1015, -72.5898, 0.7),
    "Portland":    (43.6591, -70.2568, 0.6),
    "Manchester":  (42.9956, -71.4548, 0.4),
}

# Open-Meteo variables -> our column names
WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpoint_c",
    "relative_humidity_2m": "rh_pct",
    "wind_speed_10m": "wind_kmh",
    "cloud_cover": "cloud_pct",
    # Global horizontal irradiance. Behind-the-meter rooftop solar isn't metered
    # by ISO-NE, so it shows up as LOWER demand on sunny middays; the solar
    # forecast lets the model anticipate that dip.
    "shortwave_radiation": "ghi_wm2",
}

# The Previous Runs API stores what each weather model predicted N days before
# the valid time. "previous_day1" ~= a forecast issued the day before, which
# matches our 14-38 hour lead time. Archive starts January 2024.
PREVIOUS_RUNS_START = "2024-01-01"

# Base temperature for heating/cooling degree hours (°C). 18.3 °C = 65 °F,
# the convention used in U.S. utility load analysis.
DEGREE_BASE_C = 18.3

# --- Data cleaning ---------------------------------------------------------
# ISO-NE demand has never been below ~7 GW or above ~28.2 GW (2006 record).
# Anything outside this band is a reporting error.
DEMAND_MIN_MW, DEMAND_MAX_MW = 6_000, 30_000
# Flag single-hour spikes that deviate this much from a centered 5-hour median.
SPIKE_TOLERANCE = 0.20

# --- Backtest --------------------------------------------------------------
BACKTEST_DAYS = 365      # evaluate on the most recent year (all four seasons)
RETRAIN_EVERY_DAYS = 28  # refit cadence inside the backtest
MIN_TRAIN_DAYS = 300     # don't start forecasting until we have this much history
QUANTILES = (0.1, 0.9)   # 80% prediction interval

# Where the new pipeline writes its outputs. Kept separate from the original
# data/ and models/ files so the current app keeps working during the rebuild.
DATA_DIR = "data/dayahead"
MODEL_DIR = "models/dayahead"

# --- Online calibration ----------------------------------------------------
# Applied after the models, using only forecast errors on days fully observed
# by the forecast origin (D-2 and earlier).
CALIB_WINDOW_DAYS = 28   # look-back for bias and interval calibration
BIAS_SHRINK = 0.5        # remove half the recent average error (full removal overreacts)
