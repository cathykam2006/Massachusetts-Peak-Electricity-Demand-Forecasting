"""
fair_improvements.py - pre-registered tests of two ways to improve the forecast
while using only what is known by 10 AM ET the day before (ISO-NE's day-ahead
deadline). Same model, same features, only one thing changes per test.

Test A - later morning-demand cutoff.
    Baseline: D-1 demand through the hour ending 6 AM (D1_CUTOFF_HOUR = 7).
    Candidate: through the hour ending 8 AM (D1_CUTOFF_HOUR = 9). EIA posts each
    hour ~1-2 hours after it ends, so 8 AM is in hand by about 9:45 AM.
    Judged on two years. PASS only if it lowers average error in BOTH years AND
    wins the majority of 28-day blocks in BOTH years.

Test B - freshest weather allowed.
    Baseline: weather predicted 48 h ahead (all from runs by 00 UTC on D).
    Candidate: for the target day's hours, the 06 UTC runs of GFS, ICON and ECMWF
    (published ~5-9 AM ET) and the 00 UTC run of GEM (it has no 06 UTC run), via
    Open-Meteo's Single Runs API. Those runs are only archived from April 2026,
    so this is judged on ~7 months only (models still trained on the long
    48-hour history). PASS only if it lowers average error AND wins the
    majority of blocks. Weaker evidence than Test A; reported as such.

Usage (needs internet; run on GitHub): python experiments/fair_improvements.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast import config as C                       # noqa: E402
from forecast.data import _coord_params, _get, _weighted_frame, blend_models  # noqa: E402
from forecast.features import build_features, feature_columns, modeling_rows  # noqa: E402
from forecast.models import fit_models, predict        # noqa: E402

HOURLY = f"{C.DATA_DIR}/hourly.csv"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
# model -> (run hour UTC, Open-Meteo ids to try in order)
FRESH_RUNS = {
    "gfs": (6, ["gfs_seamless", "gfs_global", "gfs025", "ncep_gfs025"]),
    "ecmwf": (6, ["ecmwf_ifs025", "ecmwf_ifs"]),
    "icon": (6, ["icon_seamless", "icon_global"]),
    "gem": (0, ["gem_seamless", "gem_global"]),
}
WX_COLS = list(C.WEATHER_VARS.values())
SPREAD = ["temp_spread", "cloud_spread", "ghi_spread"]
OUT = "experiment_results.json"


def backtest(hourly, start, end, cutoff=None):
    if cutoff is not None:
        C.D1_CUTOFF_HOUR = cutoff
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
        out.append(pd.DataFrame({"pred": p["pred"].values, "actual": te[C.TARGET].values,
                                 "iso": te["iso_forecast_mw"].values,
                                 "day": te["target_date"].values, "block": bs}, index=te.index))
    C.D1_CUTOFF_HOUR = 7
    return pd.concat(out)


def score(bt):
    bt = bt.dropna(subset=["actual"])
    ape = (bt.pred - bt.actual).abs() / bt.actual * 100
    g = bt.groupby("day")
    peak = ((g.pred.max() - g.actual.max()).abs() / g.actual.max() * 100).mean()
    both = bt.dropna(subset=["iso"])
    iso = ((both.iso - both.actual).abs() / both.actual * 100).mean()
    gi = both.groupby("day")
    iso_peak = ((gi.iso.max() - gi.actual.max()).abs() / gi.actual.max() * 100).mean()
    ours_same = ((both.pred - both.actual).abs() / both.actual * 100).mean()
    return {"hourly": ape.mean(), "peak": peak, "iso_hourly": iso, "iso_peak": iso_peak,
            "ours_on_iso_hours": ours_same, "blocks": ape.groupby(bt.block).mean()}


def verdict(a, b):
    months = int((b["blocks"] < a["blocks"]).sum())
    ok = b["hourly"] < a["hourly"] and months > len(a["blocks"]) / 2
    return ok, months, len(a["blocks"])


def fmt(s):
    return {k: round(float(v), 3) for k, v in s.items() if k != "blocks"}


def fresh_weather(hourly, start, end):
    """Target-day weather from the freshest runs published before 10 AM ET."""
    days = pd.date_range(start, end, freq="D")
    per_model, used = [], {}
    for name, (run_h, ids) in FRESH_RUNS.items():
        model_id, frames = None, []
        for cand in ids:
            try:
                t = days[0] - pd.Timedelta(days=1)
                _get(SINGLE_RUNS_URL, {**_coord_params(), "hourly": "temperature_2m",
                                       "models": cand, "run": f"{t:%Y-%m-%d}T{run_h:02d}:00",
                                       "forecast_days": 3})
                model_id = cand
                break
            except Exception as e:
                print(f"  {name}: {cand} not available ({str(e)[:80]})")
        if model_id is None:
            print(f"  {name}: no working id; skipped")
            continue
        used[name] = f"{model_id} {run_h:02d}z"
        print(f"  {name}: using {model_id}, {run_h:02d} UTC run, {len(days)} days...", flush=True)
        for d in days:
            run_day = d - pd.Timedelta(days=1)       # issued on D for target D+1
            try:
                payload = _get(SINGLE_RUNS_URL, {**_coord_params(), "hourly": ",".join(C.WEATHER_VARS),
                                                 "models": model_id,
                                                 "run": f"{run_day:%Y-%m-%d}T{run_h:02d}:00",
                                                 "forecast_days": 3}, retries=3)
            except Exception as e:
                print(f"    {d.date()}: failed ({str(e)[:60]})")
                continue
            f = _weighted_frame(payload)
            loc = f.index.tz_convert(C.TZ)
            frames.append(f[loc.tz_localize(None).normalize() == d])
            time.sleep(0.5)
        if frames:
            per_model.append(pd.concat(frames))
    if not per_model:
        return None, used
    wx = blend_models(per_model) if len(per_model) > 1 else per_model[0]
    fresh = hourly.copy()
    cols = [c for c in WX_COLS + SPREAD if c in wx.columns and c in fresh.columns]
    idx = wx.index.intersection(fresh.index)
    fresh.loc[idx, cols] = wx.loc[idx, cols].values
    print(f"  replaced weather for {len(idx)} target-day hours")
    return fresh, used


def main():
    hourly = pd.read_csv(HOURLY, index_col="period_utc")
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    last = modeling_rows(build_features(hourly))["target_date"].max()
    results = {}

    print("TEST A: morning-demand cutoff 8 AM vs 6 AM (two years)")
    years = {"Earlier year": (last - pd.Timedelta(days=729), last - pd.Timedelta(days=365)),
             "Recent year": (last - pd.Timedelta(days=364), last)}
    passed = True
    for name, (s, e) in years.items():
        a = score(backtest(hourly, s, e, cutoff=7))
        b = score(backtest(hourly, s, e, cutoff=9))
        ok, m, n = verdict(a, b)
        passed &= ok
        results[f"A {name}"] = {"baseline": fmt(a), "candidate": fmt(b), "blocks_won": f"{m}/{n}",
                                "pass": bool(ok)}
        print(f"  {name}: baseline {a['hourly']:.3f}% / peak {a['peak']:.3f}% | candidate "
              f"{b['hourly']:.3f}% / peak {b['peak']:.3f}% | ISO {a['iso_hourly']:.3f}% / peak "
              f"{a['iso_peak']:.3f}% | won {m}/{n} -> {'PASS' if ok else 'FAIL'}", flush=True)
    results["A verdict"] = "PASS" if passed else "FAIL"
    print(f"TEST A VERDICT: {results['A verdict']}\n")
    json.dump(results, open(OUT, "w"), indent=2)

    print("TEST B: freshest weather allowed (April 2026 onward)")
    s, e = pd.Timestamp("2026-04-05"), last
    fresh, used = fresh_weather(hourly, s, e)
    results["B models"] = used
    if fresh is None:
        results["B verdict"] = "NOT RUN (no single-run data)"
    else:
        a = score(backtest(hourly, s, e))
        b = score(backtest(fresh, s, e))
        ok, m, n = verdict(a, b)
        results["B"] = {"baseline": fmt(a), "candidate": fmt(b), "blocks_won": f"{m}/{n}",
                        "pass": bool(ok)}
        results["B verdict"] = "PASS" if ok else "FAIL"
        print(f"  baseline {a['hourly']:.3f}% / peak {a['peak']:.3f}% | candidate {b['hourly']:.3f}%"
              f" / peak {b['peak']:.3f}% | ISO {a['iso_hourly']:.3f}% / peak {a['iso_peak']:.3f}%"
              f" | won {m}/{n} -> {results['B verdict']}")
    print(f"TEST B VERDICT: {results['B verdict']}")
    json.dump(results, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
