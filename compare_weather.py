"""
compare_weather.py - does a different set of weather models improve the forecast?

A pre-registered test, so the decision can't be tuned to the result:
  * Same model, same features, same hours. The ONLY change is the weather input.
  * Judged on two separate years: Oct 2024 - Oct 2025 and Oct 2025 - Oct 2026.
  * PASS only if the candidate lowers average error in BOTH years AND wins the
    majority of months in BOTH years. Otherwise keep the current setup.

Usage (needs internet):
    python compare_weather.py                # single model vs. the 4-model blend
    python compare_weather.py --add hrrr     # current blend vs. blend + NOAA HRRR
    python compare_weather.py --add nbm      # current blend vs. blend + NOAA NBM

Each weather model is downloaded once and cached in data/dayahead/wx_<model>.csv,
so later comparisons are quicker.
"""

import argparse
import os
import time

import joblib
import pandas as pd

from forecast import config as C
from forecast.data import OM_PREVIOUS_RUNS_URL, _pull_history_one, blend_models
from forecast.features import build_features, feature_columns, modeling_rows
from forecast.models import fit_models, predict

HOURLY = f"{C.DATA_DIR}/hourly.csv"
WEATHER_COLS = list(C.WEATHER_VARS.values())
SPREAD_COLS = ["temp_spread", "cloud_spread", "ghi_spread"]
# Open-Meteo has renamed some model ids over time; the first one that returns data is used.
ADDITIONS = {
    "hrrr": ["ncep_hrrr_conus", "gfs_hrrr"],
    "nbm": ["ncep_nbm_conus", "ncep_nbm"],
}


def model_frame(model_id, start, end):
    """One weather model's archived day-ahead forecast, cached on disk."""
    path = f"{C.DATA_DIR}/wx_{model_id}.csv"
    if os.path.exists(path):
        f = pd.read_csv(path, index_col=0)
        f.index = pd.to_datetime(f.index, utc=True)
        return f
    print(f"  downloading {model_id} ...")
    f = _pull_history_one(OM_PREVIOUS_RUNS_URL, "_previous_day1", start, end, model_id)
    f.to_csv(path)
    return f


def get_models(ids, start, end):
    frames, used = [], []
    for m in ids:
        try:
            f = model_frame(m, start, end)
        except Exception as e:
            print(f"  {m}: failed ({str(e)[:100]}), skipped")
            continue
        if f["temp_c"].notna().mean() > 0.5:
            frames.append(f)
            used.append(m)
        else:
            print(f"  {m}: mostly empty, skipped")
    return frames, used


def first_working(aliases, start, end):
    for m in aliases:
        frames, used = get_models([m], start, end)
        if frames:
            return frames[0], used[0]
    return None, None


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


def with_weather(base, wx):
    drop = [c for c in WEATHER_COLS + SPREAD_COLS if c in base.columns]
    return base.drop(columns=drop).join(wx, how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", choices=sorted(ADDITIONS), default=None,
                    help="test the current blend plus one more weather model")
    args = ap.parse_args()
    t0 = time.time()

    base = pd.read_csv(HOURLY, index_col="period_utc")
    base.index = pd.to_datetime(base.index, utc=True)
    start = base.index.min().strftime("%Y-%m-%d")
    end = base.index.max().strftime("%Y-%m-%d")

    bundle_path = f"{C.MODEL_DIR}/bundle.pkl"
    current = (joblib.load(bundle_path).get("weather_models") if os.path.exists(bundle_path)
               else None) or C.BLEND_MODELS

    print("Weather models in the current blend:")
    frames, used = get_models(current, start, end)
    if not frames:
        print("Could not download any current weather model. Check the internet connection.")
        return
    print(f"  using: {', '.join(used)}")

    extra_id = None
    if args.add is None:
        label_a, label_b = "single weather model", f"blend of {len(used)} models"
        a_wx, b_wx = frames[0], blend_models(frames)
    else:
        extra, extra_id = first_working(ADDITIONS[args.add], start, end)
        if extra is None:
            print(f"None of {ADDITIONS[args.add]} returned data, so there is nothing to test. "
                  "Keep the current setup.")
            return
        print(f"  candidate adds: {extra_id}")
        label_a, label_b = f"current blend ({len(used)})", f"blend + {args.add.upper()}"
        a_wx, b_wx = blend_models(frames), blend_models(frames + [extra])

    a_hourly, b_hourly = with_weather(base, a_wx), with_weather(base, b_wx)
    last = modeling_rows(build_features(base))["target_date"].max()
    years = {
        "Earlier year": (last - pd.Timedelta(days=729), last - pd.Timedelta(days=365)),
        "Recent year": (last - pd.Timedelta(days=364), last),
    }
    passed = True
    print()
    for name, (s, e) in years.items():
        print(f"{name} ({s:%b %Y} - {e:%b %Y}): backtesting both versions...")
        a_m, a_p, a_blk = score(backtest(a_hourly, s, e))
        b_m, b_p, b_blk = score(backtest(b_hourly, s, e))
        months = int((b_blk < a_blk).sum())
        ok = (b_m < a_m) and (months > len(a_blk) / 2)
        passed &= ok
        print(f"  {label_a:22s}: {a_m:.2f}% hourly error, {a_p:.2f}% daily-peak error")
        print(f"  {label_b:22s}: {b_m:.2f}% hourly error, {b_p:.2f}% daily-peak error")
        print(f"  candidate better in {months} of {len(a_blk)} months -> {'PASS' if ok else 'FAIL'}\n")

    print("=" * 60)
    if passed:
        models = used + ([extra_id] if extra_id else [])
        print(f"VERDICT: PASS. Send this result to Claude before switching. Models: {', '.join(models)}")
    else:
        print("VERDICT: FAIL. Keep the current setup; no further action needed.")
    print(f"(finished in {(time.time() - t0) / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
