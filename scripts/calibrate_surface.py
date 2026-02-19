"""Calibrate parametric vol models on stored surface data."""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate SABR/Heston/SVI models on stored options chains"
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker to calibrate (default: SPY)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sabr", "svi"],
        choices=["sabr", "heston", "svi"],
        help="Models to calibrate (default: sabr svi)",
    )
    parser.add_argument("--spot", type=float, default=None, help="Override spot price")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    args = parser.parse_args()

    from src.data.storage import OptionsStorage
    from src.models.calibration import calibrate_surface_snapshot

    storage = OptionsStorage()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    dates = storage.list_dates(args.ticker)
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        logger.error("No data found for ticker %s in the specified range.", args.ticker)
        return

    logger.info("Calibrating %d snapshots for %s", len(dates), args.ticker)

    for dt in dates:
        chain = storage.load_chain(args.ticker, as_of=dt)
        if chain.empty:
            continue

        # Build a simple grid for calibration input
        spot = args.spot or chain["spot_price"].iloc[0] if "spot_price" in chain.columns else 100.0

        # Use yfinance IV directly as the "surface" for quick calibration demo
        iv_col = "implied_vol_yf"
        if iv_col not in chain.columns:
            continue

        chain = chain[chain[iv_col].notna() & (chain[iv_col] > 0)].copy()
        if len(chain) < 5:
            continue

        chain["log_moneyness"] = np.log(chain["strike"] / spot)
        chain["tenor_years"] = chain["days_to_expiry"] / 365.0

        k_arr = chain["log_moneyness"].values
        T_arr = chain["tenor_years"].values
        iv_arr = chain[iv_col].values

        # Build a minimal grid
        moneyness = np.unique(np.round(k_arr, 2))
        tenors = np.unique(np.round(T_arr, 4))
        grid = pd.DataFrame(np.nan, index=moneyness, columns=tenors)

        for i, row in chain.iterrows():
            k = round(row["log_moneyness"], 2)
            T = round(row["tenor_years"], 4)
            if k in grid.index and T in grid.columns:
                grid.loc[k, T] = row[iv_col]

        results = calibrate_surface_snapshot(
            grid, spot=float(spot), risk_free_rate=args.rate,
            models=tuple(args.models),
        )
        logger.info("Date %s: %s", dt, {m: f"RMSE={r.get('rmse', 'N/A'):.6f}" for m, r in results.items()})


if __name__ == "__main__":
    main()
