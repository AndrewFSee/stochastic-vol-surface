"""One-time backfill: load Kaggle SPY IV dataset into Parquet storage."""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Kaggle SPY IV dataset into Parquet storage"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the Kaggle CSV file",
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker to stamp the data with (default: SPY)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    from src.data.kaggle_loader import load_kaggle_spy
    from src.data.storage import OptionsStorage

    logger.info("Loading Kaggle dataset from %s", args.path)
    df = load_kaggle_spy(args.path, ticker=args.ticker, start=args.start, end=args.end)

    if df.empty:
        logger.error("No data loaded from %s", args.path)
        sys.exit(1)

    storage = OptionsStorage()
    # Save per day to partition correctly
    for dt, group in df.groupby(df["as_of_date"].dt.date):
        storage.save_chain(group, ticker=args.ticker, as_of=dt)

    logger.info("Backfill complete: %d rows for %s", len(df), args.ticker)


if __name__ == "__main__":
    main()
