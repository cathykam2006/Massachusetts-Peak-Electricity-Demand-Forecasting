"""
train_dayahead.py — build the dataset, backtest honestly, train final models.

Usage (run locally; needs an EIA API key):
    export EIA_API_KEY="your_key_here"
    python train_dayahead.py                       # 2024-01-01 -> yesterday, real weather forecasts
    python train_dayahead.py --skip-backtest       # just refit the final models (monthly retrain)
    python train_dayahead.py --weather archive --start 2021-01-01
        # perfect-hindsight weather: an UPPER BOUND on accuracy, not a forecast.
        # Comparing the two runs shows how much weather-forecast error costs.

Outputs (in data/dayahead/ and models/dayahead/):
    hourly.csv              raw hourly inputs
    backtest.csv            every out-of-sample forecast, with intervals
    metrics.csv             model comparison incl. ISO-NE's own forecast
    season_metrics.csv      accuracy by season
    shap_values.npy + shap_features.csv  (explainability for the app)
    models/dayahead/bundle.pkl           models + feature list + calibration
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from forecast import config as C
from forecast.calibration import calibration_state
from forecast.data import build_hourly_frame
from forecast.evaluate import by_season, rolling_backtest, summarize
from forecast.features import build_features, feature_columns, modeling_rows
from forecast.models import fit_models


def main():
    ap = argparse.ArgumentParser()
    yesterday = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ap.add_argument("--start", default=C.PREVIOUS_RUNS_START)
    ap.add_argument("--end", default=yesterday)
    ap.add_argument("--weather", choices=["forecast", "archive"], default="forecast")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--weather-blend", action="store_true",
                    help="blend several weather models (run compare_weather.py first)")
    ap.add_argument("--no-weather-blend", action="store_true",
                    help="switch back to the single default weather model")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set the EIA_API_KEY environment variable first.")
    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    suffix = "" if args.weather == "forecast" else "_perfect_weather"

    # 1. Data ------------------------------------------------------------
    # Keep whatever the deployed model uses unless told otherwise, so the monthly
    # automatic retrain never silently switches the weather setup.
    weather_models = None
    if args.weather_blend:
        weather_models = C.BLEND_MODELS
    elif not args.no_weather_blend and os.path.exists(f"{C.MODEL_DIR}/bundle.pkl"):
        weather_models = joblib.load(f"{C.MODEL_DIR}/bundle.pkl").get("weather_models")
    print(f"Weather: {'blend of ' + ', '.join(weather_models) if weather_models else 'single default model'}")
    hourly = build_hourly_frame(args.start, args.end, api_key, weather_mode=args.weather,
                                weather_models=weather_models)
    hourly.to_csv(f"{C.DATA_DIR}/hourly{suffix}.csv")
    print(f"Hourly frame: {hourly.shape}, {hourly[C.TARGET].isna().mean():.1%} target missing")

    feat = build_features(hourly)
    cols = feature_columns(feat)
    rows = modeling_rows(feat)

    # 2. Backtest --------------------------------------------------------
    fallback = (0.0, 0.0)
    if not args.skip_backtest:
        print("\nRolling-origin backtest...")
        bt = rolling_backtest(feat)
        bt.to_csv(f"{C.DATA_DIR}/backtest{suffix}.csv")
        table, coverage = summarize(bt)
        table.to_csv(f"{C.DATA_DIR}/metrics{suffix}.csv")
        seasons = by_season(bt)
        seasons.to_csv(f"{C.DATA_DIR}/season_metrics{suffix}.csv")
        print("\n" + table.to_string())
        print("\nBy season (calibrated ensemble):\n" + seasons.to_string())
        # Calibration to use until the live forecast log has its own history
        fallback = calibration_state(bt, bt["target_date"].max() + pd.Timedelta(days=2))
        with open(f"{C.DATA_DIR}/backtest_summary{suffix}.json", "w") as f:
            json.dump({"interval_coverage_pct": round(coverage, 1),
                       "backtest_start": str(bt["target_date"].min().date()),
                       "backtest_end": str(bt["target_date"].max().date()),
                       "weather_mode": args.weather}, f, indent=2)

    if args.weather == "archive":
        print("\nPerfect-weather run: skipping final model (not deployable).")
        return

    # 3. Final models on all data ----------------------------------------
    print("\nFitting final models on all available data...")
    models = fit_models(rows[cols], rows[C.TARGET])
    bundle = {
        "models": models,
        "weather_models": weather_models if "temp_spread" in hourly.columns else None,
        "feature_cols": cols,
        "trained_through": str(rows["target_date"].max().date()),
        "model_version": C.MODEL_VERSION,
        "fallback_calibration": {"bias": fallback[0], "q": fallback[1]},
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    joblib.dump(bundle, f"{C.MODEL_DIR}/bundle.pkl")

    # 4. SHAP (explainability for the app) -------------------------------
    try:
        from forecast.explain import save_shap
        save_shap(models["lgb"], rows, cols)
    except Exception as e:   # explanations are optional; never fail training over them
        print(f"Skipping explainability artifacts: {e}")

    print(f"\nDone. Models trained through {bundle['trained_through']}.")


if __name__ == "__main__":
    main()
