"""
models.py — point and interval models.

Point forecast: average of LightGBM and XGBoost (the ensemble that won in the
original project). Interval: LightGBM quantile regression at the 10th and 90th
percentiles, giving an 80% prediction interval an operator can plan around.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C

LGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42,
)


def fit_models(X, y):
    models = {
        "lgb": lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y),
        "xgb": xgb.XGBRegressor(**XGB_PARAMS).fit(X, y),
    }
    for q in C.QUANTILES:
        models[f"q{int(q * 100)}"] = lgb.LGBMRegressor(
            objective="quantile", alpha=q, **LGB_PARAMS
        ).fit(X, y)
    return models


def predict(models, X):
    out = pd.DataFrame(index=X.index)
    out["pred_lgb"] = models["lgb"].predict(X)
    out["pred_xgb"] = models["xgb"].predict(X)
    out["pred"] = (out["pred_lgb"] + out["pred_xgb"]) / 2
    lo, hi = (f"q{int(q * 100)}" for q in C.QUANTILES)
    # Quantile models are fit separately and can cross; enforce lo <= pred <= hi
    out["pred_lo"] = np.minimum(models[lo].predict(X), out["pred"])
    out["pred_hi"] = np.maximum(models[hi].predict(X), out["pred"])
    return out
