#!/usr/bin/env python
"""CLI: download earnings dates (past and scheduled) into data/events/earnings.parquet.

Funds and indices have none and are skipped.  The daily job refreshes these
too, so this is only needed once, or after adding tickers.

Usage:
    python scripts/fetch_earnings.py                 # every configured ticker
    python scripts/fetch_earnings.py -t AAPL -t NVDA
"""

import logging
import sys

import click

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")


@click.command()
@click.option("--tickers", "-t", multiple=True, default=None,
              help="Tickers (default: scraper tickers in config/default.yaml)")
@click.option("--events-dir", default="data/events", show_default=True)
def main(tickers, events_dir):
    """Fetch and store earnings dates."""
    from src.data.events import fetch_earnings, save_earnings, save_earnings_snapshot

    if not tickers:
        import yaml

        with open("config/default.yaml") as f:
            tickers = yaml.safe_load(f)["scraper"]["tickers"]
    df = fetch_earnings(tickers)
    if df.empty:
        raise SystemExit("No earnings dates returned.")
    save_earnings(df, events_dir)
    save_earnings_snapshot(df, pd_now().date(), events_dir)
    now = pd_now()
    nxt = df[df["announced"] >= now].groupby("ticker")["announced"].min()
    click.echo(f"\n{len(df)} releases for {df['ticker'].nunique()} tickers. Next scheduled:")
    click.echo(nxt.dt.strftime("%Y-%m-%d %H:%M").to_string())


def pd_now():
    import pandas as pd

    return pd.Timestamp.now(tz="America/New_York")


if __name__ == "__main__":
    main()
