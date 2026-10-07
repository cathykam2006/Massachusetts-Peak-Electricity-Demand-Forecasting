"""
evaluate.py — rolling-origin backtest and metrics.

A single train/test split tells you how one model did on one stretch of time.
A rolling-origin backtest replays history the way the forecast would really
have run: refit on everything known so far, forecast the next block of days,
step forward, repeat. Every prediction is genuinely out-of-sample.
"""

import numpy as np
import pandas as pd

from . import config as C
from .calibration import online_calibrate
from .features import feature_columns, modeling_rows
from .models import fit_models, predict


def rolling_backtest(feat, backtest_days=C.BACKTEST_DAYS,
                     retrain_every=C.RETRAIN_EVERY_DAYS,
                     min_train_days=C.MIN_TRAIN_DAYS, verbose=True):
    rows = modeling_rows(feat)
    cols = feature_columns(feat)
    dates = pd.DatetimeIndex(sorted(rows["target_date"].unique()))
    first, last = dates[0], dates[-1]
    start = max(last - pd.Timedelta(days=backtest_days - 1),
                first + pd.Timedelta(days=min_train_days))
    if start > last:
        raise ValueError("Not enough history for the requested backtest window.")

    results = []
    for block_start in pd.date_range(start, last, freq=f"{retrain_every}D"):
        block_end = min(block_start + pd.Timedelta(days=retrain_every - 1), last)
        # The earliest forecast in this block is issued on block_start - 1 day,
        # so training may only include days that were COMPLETE by then.
        train = rows[rows["target_date"] <= block_start - pd.Timedelta(days=2)]
        test = rows[(rows["target_date"] >= block_start) & (rows["target_date"] <= block_end)]
        if test.empty:
            continue
        models = fit_models(train[cols], train[C.TARGET])
        pred = predict(models, test[cols])
        keep = [C.TARGET, "ts_local", "target_date", "load_same_hour_d7"]
        if "iso_forecast_mw" in test:
            keep.append("iso_forecast_mw")
        results.append(test[keep].join(pred))
        if verbose:
            print(f"  {block_start.date()} -> {block_end.date()}  "
                  f"train={len(train):,}h  test={len(test):,}h")
    bt = pd.concat(results)
    bt = bt.rename(columns={"load_same_hour_d7": "pred_naive_week"})
    return online_calibrate(bt)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _hourly(y, p):
    m = y.notna() & p.notna()
    y, p = y[m], p[m]
    err = p - y
    return {
        "MAE (MW)": err.abs().mean(),
        "RMSE (MW)": np.sqrt((err ** 2).mean()),
        "MAPE (%)": (err.abs() / y).mean() * 100,
        "Bias (MW)": err.mean(),
    }


def _daily_peak(bt, col):
    """
    Accuracy on the number that drives capacity costs: each day's peak.
    Peak-hour hit = predicted peak hour within ±1 hour of the actual one.
    """
    d = bt[[C.TARGET, col, "ts_local", "target_date"]].dropna()
    g = d.groupby("target_date")
    actual_peak = g[C.TARGET].max()
    pred_peak = g[col].max()
    hour = d["ts_local"].map(lambda t: t.hour)
    actual_hr = d.assign(h=hour).loc[g[C.TARGET].idxmax(), ["target_date", "h"]].set_index("target_date")["h"]
    pred_hr = d.assign(h=hour).loc[g[col].idxmax(), ["target_date", "h"]].set_index("target_date")["h"]
    return {
        "Daily peak MAPE (%)": ((pred_peak - actual_peak).abs() / actual_peak).mean() * 100,
        "Peak hour ±1h (%)": ((pred_hr - actual_hr).abs() <= 1).mean() * 100,
    }


def summarize(bt):
    """Compare every forecast on the SAME hours (fair comparison)."""
    models = {
        "Seasonal naive (same hour last week)": "pred_naive_week",
        "Model (raw)": "pred_raw",
        "Model + calibration (ours)": "pred",
    }
    if "iso_forecast_mw" in bt and bt["iso_forecast_mw"].notna().mean() > 0.5:
        models["ISO-NE day-ahead forecast"] = "iso_forecast_mw"
    common = bt.dropna(subset=list(models.values()) + [C.TARGET])
    rows = {name: {**_hourly(common[C.TARGET], common[col]), **_daily_peak(common, col)}
            for name, col in models.items()}
    table = pd.DataFrame(rows).T.round(2)
    calibrated = bt["calib_q"] != 0       # skip the first week, before calibration kicks in
    b = bt[calibrated]
    cov = ((b[C.TARGET] >= b["pred_lo"]) & (b[C.TARGET] <= b["pred_hi"])).mean() * 100
    lo, hi = C.QUANTILES
    print(f"\n{int((hi - lo) * 100)}% interval empirical coverage: {cov:.1f}%  "
          f"(evaluated on {len(common):,} common hours)")
    return table, cov


def by_season(bt, col="pred"):
    season = bt["ts_local"].map(lambda t: {12: "Winter", 1: "Winter", 2: "Winter",
                                           3: "Spring", 4: "Spring", 5: "Spring",
                                           6: "Summer", 7: "Summer", 8: "Summer"}.get(t.month, "Fall"))
    out = {}
    for s, g in bt.groupby(season):
        out[s] = {**_hourly(g[C.TARGET], g[col]), **_daily_peak(g, col)}
    return pd.DataFrame(out).T.loc[lambda x: x.index.isin(["Winter", "Spring", "Summer", "Fall"])].round(2)
