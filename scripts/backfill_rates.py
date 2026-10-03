#!/usr/bin/env python
"""CLI: backfill the FRED risk-free-rate history into data/rates/rates_history.parquet.

FRED serves complete history, so this recovers rates for any day the daily
collector missed.  Safe and idempotent to re-run.

Usage:
    python scripts/backfill_rates.py --start 2026-01-01
    python scripts/backfill_rates.py --start 2026-01-01 --end 2026-07-31
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
@click.option("--start", default="2026-01-01", show_default=True,
              help="First date to fetch (YYYY-MM-DD)")
@click.option("--end", default=None, help="Last date to fetch (default: today)")
@click.option("--rates-dir", default="data/rates", show_default=True)
def main(start, end, rates_dir):
    """Download the Treasury curve from FRED and merge it into the history."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from src.data.rates import backfill_rates_history

    df = backfill_rates_history(start=start, end=end, rates_dir=rates_dir)
    if df.empty:
        click.echo("No rates returned — is FRED_API_KEY set in .env?", err=True)
        raise SystemExit(1)

    click.echo(f"Backfilled {len(df)} dates x {df.shape[1]} tenors "
               f"({df.index.min().date()} -> {df.index.max().date()})")
    click.echo(df.tail(5).to_string())


if __name__ == "__main__":
    main()
