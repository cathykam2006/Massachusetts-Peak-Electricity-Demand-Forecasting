"""
explain.py - SHAP explanations for the "What drives demand" page.

LightGBM can't compute SHAP values for linear trees, and model-agnostic SHAP
on them takes hours. So we explain a companion model: the same regularized
settings without linear trees, fit on the same data. It's checked against the
deployed model every time (R^2 printed below; about 0.995 in testing), and the
explanations are only saved if the two agree closely.

Run on its own after training:
    python -m forecast.explain
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import build_features, modeling_rows
from .models import REG_PARAMS

MIN_AGREEMENT = 0.98   # required R^2 between companion and deployed predictions


def save_shap(deployed, rows, cols):
    import shap
    companion = lgb.LGBMRegressor(**REG_PARAMS).fit(rows[cols], rows[C.TARGET])
    recent = rows[rows["target_date"] > rows["target_date"].max() - pd.Timedelta(days=365)]
    p_dep, p_comp = deployed.predict(recent[cols]), companion.predict(recent[cols])
    r2 = 1 - ((p_dep - p_comp) ** 2).sum() / ((p_dep - p_dep.mean()) ** 2).sum()
    print(f"Companion model agreement with deployed model: R^2 = {r2:.4f}")
    if r2 < MIN_AGREEMENT:
        print("Agreement too low; explanations not saved.")
        return
    np.save(f"{C.DATA_DIR}/shap_values.npy", shap.TreeExplainer(companion).shap_values(recent[cols]))
    recent[cols + [C.TARGET, "ts_local"]].to_csv(f"{C.DATA_DIR}/shap_features.csv")
    with open(f"{C.DATA_DIR}/shap_agreement.txt", "w") as f:
        f.write(f"{r2:.4f}")
    print(f"Saved SHAP values for {len(recent):,} hours.")


if __name__ == "__main__":
    bundle = joblib.load(f"{C.MODEL_DIR}/bundle.pkl")
    hourly = pd.read_csv(f"{C.DATA_DIR}/hourly.csv", index_col="period_utc")
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    rows = modeling_rows(build_features(hourly))
    save_shap(bundle["models"]["lgb"], rows, bundle["feature_cols"])
