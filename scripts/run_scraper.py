"""CLI entrypoint for the daily options chain scraper."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape daily options chains via yfinance")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT"],
        help="Tickers to scrape (default: SPY QQQ AAPL MSFT)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="As-of date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--min-oi",
        type=int,
        default=10,
        dest="min_oi",
        help="Minimum open interest filter (default: 10)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run in scheduled mode (daily at 4:30 PM ET)",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.date) if args.date else None

    if args.schedule:
        from src.data.scheduler import start_scheduler

        logger.info("Starting scheduler for tickers: %s", args.tickers)
        start_scheduler(args.tickers)
        return

    from src.data.scraper import scrape_multiple
    from src.data.storage import OptionsStorage

    storage = OptionsStorage()
    logger.info("Scraping tickers: %s", args.tickers)
    chains = scrape_multiple(args.tickers, as_of=as_of, min_open_interest=args.min_oi)

    for ticker, df in chains.items():
        if df.empty:
            logger.warning("No data for %s", ticker)
        else:
            storage.save_chain(df, ticker=ticker)
            logger.info("Saved %d rows for %s", len(df), ticker)

    logger.info("Scraping complete.")


if __name__ == "__main__":
    main()
