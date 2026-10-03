#!/usr/bin/env python
"""CLI: backfill daily OHLC for the underlyings into data/underlying/prices.parquet.

yfinance serves full daily history, so this can be re-run at any time to
fill gaps.  Realised-vol features need a lookback before the first surface
date, so start well ahead of it.  Safe and idempotent.

Usage:
    python scripts/backfill_underlying.py                      # config tickers
    python scripts/backfill_underlying.py --start 2009-06-01 -t SPY
"""

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


@click.command()
@click.option("--tickers", "-t", multiple=True, default=None,
              help="Tickers to fetch (default: scraper tickers in config/default.yaml)")
@click.option("--start", default="2025-01-01", show_default=True,
              help="First date to fetch (YYYY-MM-DD)")
@click.option("--end", default=None, help="Exclusive end date (default: today)")
@click.option("--underlying-dir", default="data/underlying", show_default=True)
def main(tickers, start, end, underlying_dir):
    """Download daily OHLC and merge it into the underlying price history."""
    from src.data.underlying import backfill_underlying_history

    if not tickers:
        import yaml

        with open("config/default.yaml") as f:
            tickers = yaml.safe_load(f)["scraper"]["tickers"]

    df = backfill_underlying_history(list(tickers), start=start, end=end,
                                     underlying_dir=underlying_dir)
    if df.empty:
        click.echo("No prices returned.", err=True)
        raise SystemExit(1)

    summary = df.reset_index().groupby("ticker")["date"].agg(["min", "max", "count"])
    click.echo(summary.to_string())


if __name__ == "__main__":
    main()
