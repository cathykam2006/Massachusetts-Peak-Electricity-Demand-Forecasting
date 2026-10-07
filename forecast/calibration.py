"""
calibration.py — online bias correction and conformal prediction intervals.

Two problems showed up in backtesting:
  1. Level drift: ISO-NE load trends down over time (behind-the-meter solar,
     efficiency), and tree models can't extrapolate a level shift, so raw
     forecasts run high. Fix: subtract a shrunk average of recent errors.
  2. Overconfident intervals: quantile models fit on training data are too
     narrow out-of-sample (~50% coverage for a nominal 80%). Fix: conformalized
     quantile regression — widen (or narrow) the band by the 80th percentile of
     how far recent actuals fell outside it.

Both use only days that are complete by the forecast origin (D-2 and earlier),
so the calibrated backtest remains genuinely out-of-sample.
"""

import numpy as np
import pandas as pd

from . import config as C


def calibration_state(history, target_date):
    """
    history: frame with C.TARGET, pred_raw, pred_lo_raw, pred_hi_raw, target_date
             (past forecasts whose actuals are known).
    Returns (bias, q) to apply to the forecast for `target_date`.
    """
    target_date = pd.Timestamp(target_date)
    newest = target_date - pd.Timedelta(days=2)
    oldest = newest - pd.Timedelta(days=C.CALIB_WINDOW_DAYS - 1)
    h = history[(history["target_date"] >= oldest) & (history["target_date"] <= newest)]
    h = h.dropna(subset=[C.TARGET, "pred_raw"])
    if len(h) < 24 * 7:          # need at least a week of errors
        return 0.0, 0.0
    bias = C.BIAS_SHRINK * (h["pred_raw"] - h[C.TARGET]).mean()
    lo, hi = C.QUANTILES
    score = np.maximum(h["pred_lo_raw"] - h[C.TARGET], h[C.TARGET] - h["pred_hi_raw"])
    q = float(np.quantile(score, hi - lo))
    return float(bias), q


def apply_calibration(df, bias, q):
    out = df.copy()
    out["pred"] = out["pred_raw"] - bias
    out["pred_lo"] = np.minimum(out["pred_lo_raw"] - bias - q, out["pred"])
    out["pred_hi"] = np.maximum(out["pred_hi_raw"] - bias + q, out["pred"])
    return out


def online_calibrate(bt):
    """Calibrate every day of a backtest using only errors known at its origin."""
    bt = bt.rename(columns={"pred": "pred_raw", "pred_lo": "pred_lo_raw",
                            "pred_hi": "pred_hi_raw"})
    parts = []
    for d, day in bt.groupby("target_date"):
        bias, q = calibration_state(bt, d)
        parts.append(apply_calibration(day, bias, q).assign(calib_bias=bias, calib_q=q))
    return pd.concat(parts).sort_index()
