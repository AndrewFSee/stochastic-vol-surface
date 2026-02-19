#!/usr/bin/env python
"""CLI: run the walk-forward backtest."""

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--ticker", "-t", default="SPY", show_default=True)
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--strategy", default="straddle",
              type=click.Choice(["straddle", "risk_reversal", "butterfly"]))
@click.option("--capital", default=1_000_000.0, show_default=True)
@click.option("--output", default="backtest_results.csv", show_default=True)
def main(ticker, start, end, strategy, capital, output):
    """Run walk-forward backtest and save results to CSV."""
    import pandas as pd
    from datetime import date
    from src.backtest.engine import run_backtest, BacktestConfig
    from src.backtest.strategies import (
        delta_hedged_straddle, risk_reversal, butterfly_trade,
    )
    from src.backtest.metrics import compute_all_metrics
    from src.data.storage import load_options_chain

    # Build mock surface history from stored data
    df = load_options_chain(ticker)
    if df.empty:
        click.echo(f"No data found for {ticker}", err=True)
        raise SystemExit(1)

    # Placeholder: build VolSurface objects from data
    # (Full implementation requires calibrated surfaces on disk)
    click.echo("Note: full backtest requires pre-built surfaces. Running demo.")

    config = BacktestConfig(initial_capital=capital)
    strategy_map = {
        "straddle": delta_hedged_straddle,
        "risk_reversal": risk_reversal,
        "butterfly": butterfly_trade,
    }
    strategy_fn = strategy_map[strategy]

    # Demo: use empty surface history
    results = run_backtest({}, strategy_fn=strategy_fn, config=config)
    if results.empty:
        click.echo("No backtest results generated (empty surface history).")
        return

    results.to_csv(output, index=False)
    metrics = compute_all_metrics(results)
    click.echo(f"Results saved to {output}")
    click.echo(f"Sharpe: {metrics['sharpe']:.2f}  MaxDD: {metrics['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
