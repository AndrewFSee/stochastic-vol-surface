#!/usr/bin/env python
"""CLI: calibrate all parametric vol models for a date range."""

import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--ticker", "-t", default="SPY", show_default=True)
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--data-dir", default="data/options", show_default=True)
@click.option("--output-dir", default="data/models", show_default=True)
def main(ticker, start, end, data_dir, output_dir):
    """Calibrate SABR, Heston, and SVI for each available date."""
    import pandas as pd
    import json
    from pathlib import Path
    from src.data.storage import load_options_chain
    from src.models.sabr import calibrate_sabr
    from src.models.svi import calibrate_svi
    import numpy as np

    df = load_options_chain(ticker, base_dir=data_dir, start=start, end=end)
    if df.empty:
        click.echo(f"No data found for {ticker} in {data_dir} "
                   f"between {start} and {end}", err=True)
        raise SystemExit(1)

    df["as_of"] = pd.to_datetime(df["as_of"])

    # The scraper stores the market-quoted IV under `implied_volatility_market`;
    # the Kaggle loader normalises to `implied_volatility`.  Accept either.
    iv_col = next(
        (c for c in ("implied_volatility", "implied_volatility_market")
         if c in df.columns),
        None,
    )
    if iv_col is None:
        click.echo("Chain has no implied-volatility column.", err=True)
        raise SystemExit(1)

    from src.data.rates import get_rate_for_tenor, load_rates_history
    from src.surface.grid_builder import default_k_grid

    rates_history = load_rates_history()

    k_nodes = default_k_grid()
    k_min, k_max = float(k_nodes.min()), float(k_nodes.max())

    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for date, group in df.groupby("as_of"):
        # Use 3M expiry, call options with IV
        slice_df = group[
            (group["option_type"] == "call")
            & (group["T"].between(0.20, 0.35))
            & group[iv_col].notna()
            & (group[iv_col] > 0)
        ].sort_values("strike")

        if len(slice_df) < 5:
            continue

        # Median, not first: a scrape spanning a price move stores several
        # distinct underlying prices in one snapshot.
        spot_series = slice_df.get("underlying_price", pd.Series(dtype=float)).dropna()
        spot = float(spot_series.median()) if not spot_series.empty else 100.0
        T = float(slice_df["T"].median())

        # Calibrate on the forward, matching the surface layer's convention.
        r = get_rate_for_tenor(date.date(), T, history=rates_history)
        F = spot * np.exp(r * T)

        strikes = slice_df["strike"].to_numpy()
        ivs = slice_df[iv_col].to_numpy()

        # Restrict to the same log-moneyness band the surface grid uses.  The
        # raw chain runs far into the wings, where stale quotes carry IVs an
        # order of magnitude off; fitting those wrecks the smile calibration.
        k_all = np.log(strikes / F)
        band = (k_all >= k_min) & (k_all <= k_max)
        if band.sum() < 5:
            logger.warning("%s %s: only %d quotes in [%.2f, %.2f] — skipped",
                           ticker, date.date(), int(band.sum()), k_min, k_max)
            continue
        strikes, ivs = strikes[band], ivs[band]

        sabr_params = calibrate_sabr(F, strikes, ivs, T)
        k = np.log(strikes / F)
        w = ivs ** 2 * T
        svi_params = calibrate_svi(k, w)

        row = {"date": str(date.date()), "ticker": ticker,
               "spot": spot, "forward": float(F), "r": float(r), "T": T,
               "n_quotes": len(slice_df),
               "sabr": sabr_params, "svi": svi_params}
        results.append(row)
        logger.info("Calibrated %s %s: SABR rmse=%.4f SVI rmse=%.4f",
                    ticker, date.date(), sabr_params["rmse"], svi_params["rmse"])

    out_file = out_dir / f"calibration_{start}_{end}.json"
    out_file.write_text(json.dumps(results, indent=2))
    click.echo(f"Wrote {len(results)} calibration records to {out_file}")


if __name__ == "__main__":
    main()
