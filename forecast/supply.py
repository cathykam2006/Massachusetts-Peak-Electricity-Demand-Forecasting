"""
supply.py - the supply side: what generated New England's electricity each hour.

Source: EIA-930 hourly data for ISO-NE (respondent "ISNE").
  * fuel-type-data: net generation by energy source (MWh in each hour = average MW)
  * region-data, type "TI": total interchange. EIA's sign convention is
    positive = net exports, so net imports = -TI.

Hourly balance (approximately): demand = net generation + net imports.
Battery and pumped storage are negative while charging; they are reported as
their own group so the stack stays honest.
"""

import pandas as pd

from . import config as C
from .data import _get, pull_eia_series

EIA_FUEL_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
SUPPLY_PATH = f"{C.DATA_DIR}/supply.csv"
KEEP_DAYS = 400

# EIA fuel codes -> the groups shown in the app. Unknown codes fall into "Other".
FUEL_GROUPS = {
    "NUC": "Nuclear",
    "WAT": "Hydro",
    "SUN": "Solar",
    "NG": "Natural gas",
    "WND": "Wind",
    "OIL": "Oil & other",
    "COL": "Oil & other",
    "OTH": "Oil & other",
    "GEO": "Oil & other",
    "UNK": "Oil & other",
    "BAT": "Storage",
    "PS": "Storage",
    "SNB": "Storage",
}
GROUP_ORDER = ["Nuclear", "Hydro", "Solar", "Natural gas", "Wind", "Net imports",
               "Oil & other", "Storage"]


def pull_fuel_mix(start, end, api_key):
    """Hourly net generation by group (MW), UTC index. start/end: 'YYYY-MM-DD'."""
    records, offset = [], 0
    while True:
        params = {
            "api_key": api_key, "frequency": "hourly", "data[0]": "value",
            "facets[respondent][]": "ISNE",
            "start": f"{start}T00", "end": f"{end}T23",
            "sort[0][column]": "period", "sort[0][direction]": "asc",
            "offset": offset, "length": 5000,
        }
        batch = _get(EIA_FUEL_URL, params)["response"]["data"]
        records.extend(batch)
        if len(batch) < 5000:
            break
        offset += 5000
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H").dt.tz_localize("UTC")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["group"] = df["fueltype"].map(FUEL_GROUPS).fillna("Oil & other")
    wide = df.pivot_table(index="period", columns="group", values="value", aggfunc="sum")
    return wide.sort_index()


def build_supply(start, end, api_key):
    """Generation by group + net imports + reported demand, hourly, UTC index."""
    mix = pull_fuel_mix(start, end, api_key)
    ti = pull_eia_series("TI", start, end, api_key)
    demand = pull_eia_series("D", start, end, api_key)
    out = mix.copy()
    out["Net imports"] = -ti.reindex(out.index)
    out["demand_mw"] = demand.reindex(out.index)
    out.index.name = "period_utc"
    cols = [g for g in GROUP_ORDER if g in out.columns] + ["demand_mw"]
    return out[cols]


def update_supply(api_key, days=None):
    """Refresh data/dayahead/supply.csv. First run pulls KEEP_DAYS; later runs
    re-pull the last 10 days (EIA revises recent hours) and merge."""
    import os
    today = pd.Timestamp.now(tz="UTC").normalize()
    exists = os.path.exists(SUPPLY_PATH)
    days = days or (10 if exists else KEEP_DAYS)
    start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    new = build_supply(start, end, api_key)
    if exists:
        old = pd.read_csv(SUPPLY_PATH, index_col="period_utc")
        old.index = pd.to_datetime(old.index, utc=True)
        new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    new = new[new.index >= today - pd.Timedelta(days=KEEP_DAYS)]
    os.makedirs(C.DATA_DIR, exist_ok=True)
    new.to_csv(SUPPLY_PATH)
    return new
