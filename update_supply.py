"""
update_supply.py - refresh the supply-side data (generation by source + imports).

    export EIA_API_KEY="your_key"
    python update_supply.py            # first run: last 400 days; later: last 10 days
    python update_supply.py --days 60  # force a specific look-back

Writes data/dayahead/supply.csv, which the app's "Where the power comes from" page reads.
"""

import argparse
import os

from forecast.supply import update_supply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("Set the EIA_API_KEY environment variable first.")
    df = update_supply(key, days=args.days)
    print(f"Supply data: {len(df):,} hours, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d %H:00} UTC")
    print("Sources:", ", ".join(c for c in df.columns if c != "demand_mw"))


if __name__ == "__main__":
    main()
