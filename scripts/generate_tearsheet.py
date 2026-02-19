"""Generate HTML tearsheet from backtest results."""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML tearsheet from backtest results")
    parser.add_argument(
        "--input",
        default="data/calibrations/backtest_returns.parquet",
        help="Path to backtest returns Parquet file",
    )
    parser.add_argument(
        "--output",
        default="reports/backtest.html",
        help="Output HTML path (default: reports/backtest.html)",
    )
    parser.add_argument(
        "--title",
        default="Vol-Arb Strategy Backtest",
        help="Report title",
    )
    args = parser.parse_args()

    import pandas as pd
    from pathlib import Path

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.info("Run run_backtest.py first to generate backtest results.")
        return

    returns = pd.read_parquet(input_path)
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    from src.backtest.tearsheet import generate_tearsheet

    out = generate_tearsheet(returns, output_path=args.output, title=args.title)
    logger.info("Tearsheet saved to: %s", out)


if __name__ == "__main__":
    main()
