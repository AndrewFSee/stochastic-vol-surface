#!/usr/bin/env python
"""CLI: backfill full SPY end-of-day chains for 2024–2025 from Kaggle JSON.

Dataset: "S&P500 Options (SPY) Implied Volatility (2014-25)"
(shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24), one
~1 GB JSON file per year with every listed contract's bid, ask, volume,
open interest and IV.  It covers the gap between the 2010–2023 Kaggle CSV
backfill and the live collection that starts in Feb 2026.

The files carry no underlying price; the SPY close from the price store is
used (run scripts/backfill_underlying.py first).  Files are streamed one
trading day at a time, so memory stays small.

Usage:
    python scripts/backfill_kaggle_json.py --download --years 24 25
    python scripts/backfill_kaggle_json.py --download-dir D:/kaggle \\
        --output-dir data/historical/options --years 24
"""

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logging.getLogger("src.data.storage").setLevel(logging.WARNING)

DATASET = "shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24"


@click.command()
@click.option("--years", multiple=True, default=("24", "25"), show_default=True,
              help="Two-digit file years (spy_options_data_<YY>.json)")
@click.option("--download", is_flag=True, help="Download missing files from Kaggle first")
@click.option("--download-dir", default="data/kaggle", show_default=True)
@click.option("--output-dir", default="data/historical/options", show_default=True)
@click.option("--underlying-dir", default="data/underlying", show_default=True)
def main(years, download, download_dir, output_dir, underlying_dir):
    """Normalise the JSON chains into the partitioned Parquet store."""
    from src.data.kaggle_loader import iter_json_days, normalise_json_records
    from src.data.storage import save_options_chain
    from src.data.underlying import load_underlying_history

    prices = load_underlying_history(underlying_dir, tickers=["SPY"])
    if prices.empty:
        raise SystemExit("No SPY prices; run scripts/backfill_underlying.py -t SPY first.")
    spot = prices.xs("SPY", level="ticker")["close"]

    dl = Path(download_dir)
    files = [dl / f"spy_options_data_{y}.json" for y in years]
    missing = [f for f in files if not f.exists()]
    if missing and download:
        from dotenv import load_dotenv
        from kaggle.api.kaggle_api_extended import KaggleApi

        load_dotenv()
        api = KaggleApi()
        api.authenticate()
        dl.mkdir(parents=True, exist_ok=True)
        for f in missing:
            click.echo(f"downloading {f.name} ...")
            api.dataset_download_file(DATASET, f.name, path=str(dl), quiet=True)
    elif missing:
        raise SystemExit(f"Missing {', '.join(map(str, missing))}; use --download.")

    for f in files:
        days = rows = 0
        for records in iter_json_days(f):
            chain = normalise_json_records(records, spot)
            if chain.empty:
                continue
            save_options_chain(chain, output_dir, validate=False)
            days += chain["as_of"].nunique()
            rows += len(chain)
        click.echo(f"{f.name}: {days} days, {rows:,} rows -> {output_dir}")


if __name__ == "__main__":
    main()
