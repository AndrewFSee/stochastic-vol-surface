#!/usr/bin/env python
"""CLI: run the options-chain scraper manually (wraps scripts/scrape.py logic).

Usage:
    python scripts/run_scraper.py
    python scripts/run_scraper.py --tickers SPY QQQ
    python scripts/run_scraper.py --tickers SPY --date 2025-12-01
"""

import click
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


@click.command()
@click.option("--tickers", "-t", multiple=True,
              default=["SPY", "QQQ", "AAPL", "MSFT"],
              show_default=True, help="Tickers to scrape")
@click.option("--date", "-d", default=None,
              help="As-of date YYYY-MM-DD (default: today)")
@click.option("--output-dir", default="data/options", show_default=True)
def main(tickers, date, output_dir):
    """Scrape daily options chains via yfinance and store to Parquet."""
    from src.data.scraper import scrape_all
    from src.data.storage import save_options_chain
    import pandas as pd

    as_of = pd.Timestamp(date).date() if date else None
    click.echo(f"Scraping {list(tickers)} as of {as_of or 'today'} …")

    df = scrape_all(list(tickers), as_of=as_of)
    if df.empty:
        click.echo("No data returned — check network / market hours.", err=True)
        raise SystemExit(1)

    click.echo(f"  Fetched {len(df):,} option rows across {df['ticker'].nunique()} ticker(s)")

    # Quick sanity stats
    for tkr, grp in df.groupby("ticker"):
        n_exp = grp["expiration"].nunique() if "expiration" in grp.columns else "?"
        click.echo(f"    {tkr}: {len(grp):>6,} rows, {n_exp} expirations")

    paths = save_options_chain(df, base_dir=output_dir)
    click.echo(f"Saved to {len(paths)} partition(s) in {output_dir}/")


if __name__ == "__main__":
    main()
