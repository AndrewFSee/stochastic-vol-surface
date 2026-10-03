#!/usr/bin/env python
"""CLI: build the stored volatility forecasts (HAR + implied, pooled).

Reads   data/{historical/,}features, data/underlying/prices.parquet
Writes  data/forecasts/vol_forecasts.parquet  (one row per ticker x date x horizon)

Every row is an out-of-sample, point-in-time forecast (walk-forward, monthly
refits), so the history doubles as the model's track record and as an ML
feature.  Rebuilt in full each run.

Usage:
    python scripts/build_forecasts.py
"""

import logging
import sys
import warnings

import click

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")


@click.command()
@click.option("--out", default="data/forecasts/vol_forecasts.parquet", show_default=True)
@click.option("--start", default="2013-01-01", show_default=True)
def main(out, start):
    """Build and save forecasts for every ticker in the feature tables."""
    from src.forecast.forecaster import build_forecasts, save_forecasts

    warnings.filterwarnings("ignore")
    df = build_forecasts(start=start)
    if df.empty:
        click.echo("No forecasts built (missing features or prices?).", err=True)
        raise SystemExit(1)
    save_forecasts(df, out)
    latest = df[df["date"] == df["date"].max()]
    click.echo(latest[["ticker", "horizon", "forecast_vol", "lo80_vol", "hi80_vol",
                       "implied_vol", "implied_input"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
