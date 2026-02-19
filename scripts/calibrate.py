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

    df = load_options_chain(ticker, base_dir=data_dir)
    if df.empty:
        click.echo(f"No data found for {ticker} in {data_dir}", err=True)
        raise SystemExit(1)

    df["as_of"] = pd.to_datetime(df["as_of"])
    mask = (df["as_of"] >= start) & (df["as_of"] <= end)
    df = df[mask]

    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for date, group in df.groupby("as_of"):
        # Use 3M expiry, call options with IV
        slice_df = group[
            (group["option_type"] == "call")
            & (group["T"].between(0.20, 0.35))
            & group["implied_volatility"].notna()
        ].sort_values("strike")

        if len(slice_df) < 5:
            continue

        spot = slice_df.get("underlying_price", pd.Series([100.0])).iloc[0]
        F = float(spot) if pd.notna(spot) else 100.0
        strikes = slice_df["strike"].to_numpy()
        ivs = slice_df["implied_volatility"].to_numpy()
        T = float(slice_df["T"].median())

        sabr_params = calibrate_sabr(F, strikes, ivs, T)
        k = np.log(strikes / F)
        w = ivs ** 2 * T
        svi_params = calibrate_svi(k, w)

        row = {"date": str(date.date()), "ticker": ticker,
               "sabr": sabr_params, "svi": svi_params}
        results.append(row)
        logger.info("Calibrated %s %s: SABR rmse=%.4f SVI rmse=%.4f",
                    ticker, date.date(), sabr_params["rmse"], svi_params["rmse"])

    out_file = out_dir / f"calibration_{start}_{end}.json"
    out_file.write_text(json.dumps(results, indent=2))
    click.echo(f"Wrote {len(results)} calibration records to {out_file}")


if __name__ == "__main__":
    main()
