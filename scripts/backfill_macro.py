#!/usr/bin/env python
"""CLI: backfill FRED macro and credit series into data/macro/macro_history.parquet.

Usage:
    python scripts/backfill_macro.py --start 2009-01-01
"""

import logging
import sys

import click

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")


@click.command()
@click.option("--start", default="2009-01-01", show_default=True)
@click.option("--end", default=None)
@click.option("--macro-dir", default="data/macro", show_default=True)
def main(start, end, macro_dir):
    """Download the macro series and merge them into the history."""
    from dotenv import load_dotenv

    load_dotenv()
    from src.data.macro import fetch_macro, save_macro_history

    df = fetch_macro(start, end)
    if df.empty:
        raise SystemExit("No macro data returned (FRED_API_KEY set?)")
    save_macro_history(df, macro_dir)
    click.echo(df.notna().sum().to_string())


if __name__ == "__main__":
    main()
