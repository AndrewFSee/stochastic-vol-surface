#!/usr/bin/env python
"""CLI: backfill VIX-family index history (VIX, VIX3M, VIX9D, SKEW, VVIX).

The daily collector only stores a rolling 5-day window from the day it runs,
so history before collection began needs this.  Writes
data/vix/vix_0000_backfill.parquet; daily snapshots take precedence over it.

Usage:
    python scripts/backfill_vix.py --start 2009-06-01
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
@click.option("--start", default="2009-06-01", show_default=True, help="First date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Exclusive end date (default: today)")
@click.option("--vix-dir", default="data/vix", show_default=True)
def main(start, end, vix_dir):
    """Download VIX-family history and save it as the backfill file."""
    from src.data.vix_family import backfill_vix_history

    df = backfill_vix_history(start, end, vix_dir)
    if df.empty:
        raise SystemExit(1)
    click.echo(df.notna().sum().to_string())


if __name__ == "__main__":
    main()
