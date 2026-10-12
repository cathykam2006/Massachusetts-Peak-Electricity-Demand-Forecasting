"""
compare_weather.py - does blending several weather models improve the forecast?

A pre-registered test, so the decision can't be tuned to the result:
  * Same model, same features, same hours. The ONLY change is the weather input:
    the default single model vs. a blend of BLEND_MODELS (mean + spread).
  * Judged on two separate years: Oct 2024 - Oct 2025 and Oct 2025 - Oct 2026.
  * PASS only if the blend lowers average error in BOTH years AND wins the
    majority of months in BOTH years. Otherwise keep the single model.

Usage (needs internet; about 10-15 minutes):
    python compare_weather.py
"""

import os
import time

import numpy as np
import pandas as pd

from forecast import config as C
from forecast.data import pull_weather_history
from forecast.features import build_features, feature_columns, modeling_rows
from forecast.models import fit_models, predict

HOURLY = f"{C.DATA_DIR}/hourly.csv"
BLEND_CACHE = f"{C.DATA_DIR}/weather_blend.csv"
WEATHER_COLS = list(C.WEATHER_VARS.values())


def backtest(hourly, start, end):
    feat = build_features(hourly)
    rows, cols = modeling_rows(feat), feature_columns(feat)
    out = []
    for bs in pd.date_range(start, end, freq=f"{C.RETRAIN_EVERY_DAYS}D"):
        tr = rows[rows["target_date"] <= bs - pd.Timedelta(days=2)]
        te = rows[(rows["target_date"] >= bs) &
                  (rows["target_date"] <= min(bs + pd.Timedelta(days=C.RETRAIN_EVERY_DAYS - 1), end))]
        if te.empty:
            continue
        p = predict(fit_models(tr[cols], tr[C.TARGET]), te[cols])
        out.append(pd.DataFrame({"pred": p["pred"], "actual": te[C.TARGET],
                                 "day": te["target_date"], "block": bs}))
    return pd.concat(out)


def score(bt):
    ape = (bt["pred"] - bt["actual"]).abs() / bt["actual"] * 100
    g = bt.groupby("day")
    peak = ((g["pred"].max() - g["actual"].max()).abs() / g["actual"].max() * 100).mean()
    return ape.mean(), peak, ape.groupby(bt["block"]).mean()


def main():
    t0 = time.time()
    single = pd.read_csv(HOURLY, index_col="period_utc")
    single.index = pd.to_datetime(single.index, utc=True)
    start = single.index.min().strftime("%Y-%m-%d")
    end = single.index.max().strftime("%Y-%m-%d")

    if os.path.exists(BLEND_CACHE):
        wx = pd.read_csv(BLEND_CACHE, index_col=0)
        wx.index = pd.to_datetime(wx.index, utc=True)
        print(f"Using cached blended weather ({BLEND_CACHE}).")
    else:
        print(f"Pulling blended weather {start} to {end} "
              f"({', '.join(C.BLEND_MODELS)}); this takes a few minutes...")
        wx = pull_weather_history(start, end, mode="forecast", models=C.BLEND_MODELS)
        wx.to_csv(BLEND_CACHE)
    if "temp_spread" not in wx.columns:
        print("Only one weather model came back, so there is nothing to compare. Keep the single model.")
        return

    blend = single.drop(columns=[c for c in WEATHER_COLS if c in single.columns]).join(wx, how="left")

    last = modeling_rows(build_features(single))["target_date"].max()
    years = {
        "Earlier year": (last - pd.Timedelta(days=729), last - pd.Timedelta(days=365)),
        "Recent year": (last - pd.Timedelta(days=364), last),
    }
    passed = True
    print()
    for name, (s, e) in years.items():
        print(f"{name} ({s:%b %Y} - {e:%b %Y}): backtesting both versions...")
        a_m, a_p, a_blk = score(backtest(single, s, e))
        b_m, b_p, b_blk = score(backtest(blend, s, e))
        months = int((b_blk < a_blk).sum())
        ok = (b_m < a_m) and (months > len(a_blk) / 2)
        passed &= ok
        print(f"  single weather model : {a_m:.2f}% hourly error, {a_p:.2f}% daily-peak error")
        print(f"  blended weather      : {b_m:.2f}% hourly error, {b_p:.2f}% daily-peak error")
        print(f"  blend better in {months} of {len(a_blk)} months -> {'PASS' if ok else 'FAIL'}\n")

    print("=" * 60)
    if passed:
        print("VERDICT: PASS. The blend helps in both years. Switch it on with:")
        print("    python train_dayahead.py --weather-blend")
    else:
        print("VERDICT: FAIL. The blend did not help consistently. Keep the current model;")
        print("no further action needed.")
    print(f"(finished in {(time.time() - t0) / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
