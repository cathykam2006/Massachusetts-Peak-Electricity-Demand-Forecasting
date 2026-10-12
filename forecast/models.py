"""
models.py — point and interval models.

Point forecast: one regularized LightGBM model with linear trees.
  * Regularized (few leaves, large minimum leaf size, L2 penalty) because the
    original 63-leaf model fit training data to ~1% error but new days to ~4%,
    a clear sign of overfitting.
  * Linear trees: each leaf fits a small linear model instead of a constant,
    so the forecast can extend the temperature trend on record cold snaps and
    heat waves instead of flattening at the edge of the training data.
  This configuration was fixed in advance from standard practice, judged on a
  year not used to choose it (Oct 2024-Oct 2025), then confirmed on the next
  year: lower error in 11 of 13 and 10 of 14 monthly blocks, and roughly half
  the train-test gap.

Peak: a daily-peak model averaged with the hourly model's peak (see below).

Interval: LightGBM quantile regression at the 10th and 90th percentiles with
the same regularization, then conformally calibrated (see calibration.py).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

REG_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=100,
    reg_lambda=5.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)
POINT_PARAMS = {**REG_PARAMS, "linear_tree": True, "linear_lambda": 5.0}


# Daily-peak model (passed a pre-registered two-year test: daily-peak error
# 2.69% -> 2.51% and 2.14% -> 2.03%, better in 11 of 14 and 10 of 13 months, with
# hourly error unchanged). One row per day: the day's 6 PM feature row, which
# carries day-level weather, calendar and recent peaks. Target: the day's peak.
PEAK_FEATURE_HOUR = 18
PEAK_WEIGHT = 0.5        # final peak = average of the hourly and peak models
PEAK_TAPER_HOURS = 4     # the shift fades to zero 4 hours either side of the peak
PEAK_PARAMS = {**POINT_PARAMS, "min_child_samples": 20}   # ~1,000 daily rows


def _local_day(index):
    return pd.DatetimeIndex(index.tz_convert(C.TZ).date)


def fit_models(X, y):
    models = {"lgb": lgb.LGBMRegressor(**POINT_PARAMS).fit(X, y)}
    for q in C.QUANTILES:
        models[f"q{int(q * 100)}"] = lgb.LGBMRegressor(
            objective="quantile", alpha=q, **REG_PARAMS
        ).fit(X, y)
    # Peak model: complete days only (23+ hours observed), so the peak is real
    day = pd.Series(_local_day(X.index), index=X.index)
    counts = y.groupby(day).count()
    peaks = y.groupby(day).max()[counts >= 23]
    rows = X[X["hour"] == PEAK_FEATURE_HOUR]
    rows_day = _local_day(rows.index)
    keep = rows_day.isin(peaks.index)
    if keep.sum() >= 100:
        models["peak"] = lgb.LGBMRegressor(**PEAK_PARAMS).fit(
            rows[keep], peaks.reindex(rows_day[keep]).values)
    return models


def _peak_adjustment(models, X, pred):
    """Shift each day's curve toward the combined peak estimate: fully at the
    predicted peak hour, tapering linearly to zero PEAK_TAPER_HOURS away."""
    adj = pd.Series(0.0, index=X.index)
    if "peak" not in models:
        return adj
    day = pd.Series(_local_day(X.index), index=X.index)
    rows = X[X["hour"] == PEAK_FEATURE_HOUR]
    if rows.empty:
        return adj
    peak_est = pd.Series(models["peak"].predict(rows), index=_local_day(rows.index))
    for d, idx in day.groupby(day).groups.items():
        if d not in peak_est.index:
            continue
        p = pred.loc[idx]
        delta = PEAK_WEIGHT * (peak_est.loc[d] - p.max())
        dist = (X.loc[idx, "hour"] - X.loc[p.idxmax(), "hour"]).abs()
        adj.loc[idx] = delta * np.clip(1 - dist / (PEAK_TAPER_HOURS + 1), 0, 1)
    return adj


def predict(models, X):
    out = pd.DataFrame(index=X.index)
    out["pred_lgb"] = models["lgb"].predict(X)
    adj = _peak_adjustment(models, X, out["pred_lgb"])
    out["pred"] = out["pred_lgb"] + adj
    lo, hi = (f"q{int(q * 100)}" for q in C.QUANTILES)
    # Quantile models are fit separately and can cross; enforce lo <= pred <= hi
    out["pred_lo"] = np.minimum(models[lo].predict(X) + adj, out["pred"])
    out["pred_hi"] = np.maximum(models[hi].predict(X) + adj, out["pred"])
    return out
