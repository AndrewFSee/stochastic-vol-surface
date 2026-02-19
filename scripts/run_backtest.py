"""Run the vol-arb strategy backtest on historical surfaces."""

from __future__ import annotations

import argparse
import logging
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vol-arb strategy backtest")
    parser.add_argument("--ticker", default="SPY", help="Ticker (default: SPY)")
    parser.add_argument("--strategy", default="straddle",
                        choices=["straddle", "risk_reversal", "calendar"],
                        help="Strategy to backtest")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/calibrations/backtest_returns.parquet")
    args = parser.parse_args()

    from src.data.storage import OptionsStorage
    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import compute_metrics

    storage = OptionsStorage()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    # TODO: Load VolSurface objects from storage
    # This is a placeholder that shows the backtest structure
    logger.info("Loading surface data for %s...", args.ticker)
    logger.info(
        "Backtesting strategy: %s (start=%s, end=%s)", args.strategy, start, end
    )
    logger.warning(
        "Full backtest requires pre-built VolSurface objects. "
        "Run calibrate_surface.py first to generate surfaces."
    )


if __name__ == "__main__":
    main()
