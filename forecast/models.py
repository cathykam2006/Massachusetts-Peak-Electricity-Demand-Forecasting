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


def fit_models(X, y):
    models = {"lgb": lgb.LGBMRegressor(**POINT_PARAMS).fit(X, y)}
    for q in C.QUANTILES:
        models[f"q{int(q * 100)}"] = lgb.LGBMRegressor(
            objective="quantile", alpha=q, **REG_PARAMS
        ).fit(X, y)
    return models


def predict(models, X):
    out = pd.DataFrame(index=X.index)
    out["pred_lgb"] = models["lgb"].predict(X)
    out["pred"] = out["pred_lgb"]
    lo, hi = (f"q{int(q * 100)}" for q in C.QUANTILES)
    # Quantile models are fit separately and can cross; enforce lo <= pred <= hi
    out["pred_lo"] = np.minimum(models[lo].predict(X), out["pred"])
    out["pred_hi"] = np.maximum(models[hi].predict(X), out["pred"])
    return out
