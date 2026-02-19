#!/usr/bin/env python
"""CLI: run daily options-chain scraper."""

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--tickers", "-t", multiple=True, default=["SPY", "QQQ", "AAPL", "MSFT"],
              show_default=True, help="Tickers to scrape")
@click.option("--date", "-d", default=None, help="As-of date YYYY-MM-DD (default: today)")
@click.option("--output-dir", default="data/options", show_default=True)
def main(tickers, date, output_dir):
    """Scrape daily options chains and store to Parquet."""
    from src.data.scraper import scrape_all
    from src.data.storage import save_options_chain
    import pandas as pd

    as_of = pd.Timestamp(date).date() if date else None
    click.echo(f"Scraping {list(tickers)} …")
    df = scrape_all(list(tickers), as_of=as_of)
    if df.empty:
        click.echo("No data returned.", err=True)
        raise SystemExit(1)
    paths = save_options_chain(df, base_dir=output_dir)
    click.echo(f"Saved {len(df)} rows to {len(paths)} partition(s).")


if __name__ == "__main__":
    main()
