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
              type=click.Choice(["straddle", "short_straddle", "risk_reversal", "butterfly"]))
@click.option("--capital", default=1_000_000.0, show_default=True)
@click.option("--surfaces-dir", default="data/surfaces", show_default=True)
@click.option("--tenor", default=0.25, show_default=True, help="Option tenor in years")
@click.option("--holding-days", default=5, show_default=True,
              help="Trading days to hold before rolling")
@click.option("--target-vega", default=10_000.0, show_default=True,
              help="Cash P&L per 1 vol point of parallel IV move")
@click.option("--delta-hedge/--no-delta-hedge", default=True, show_default=True)
@click.option("--output", default="backtest_results.csv", show_default=True)
def main(ticker, start, end, strategy, capital, surfaces_dir, tenor,
         holding_days, target_vega, delta_hedge, output):
    """Run walk-forward backtest over the stored surface corpus."""
    from functools import partial
    from pathlib import Path

    from experimental.backtest.engine import run_backtest, BacktestConfig
    from experimental.backtest.metrics import compute_all_metrics
    from experimental.backtest.strategies import STRATEGIES
    from src.surface.batch import load_surfaces

    surfaces = load_surfaces(ticker, surfaces_dir=surfaces_dir, start=start, end=end)
    if not surfaces:
        click.echo(
            f"No surfaces for {ticker} in {surfaces_dir} between {start} and {end}.\n"
            f"Build them first: python scripts/build_surfaces.py -t {ticker}",
            err=True,
        )
        raise SystemExit(1)

    dates = sorted(surfaces)
    click.echo(f"Loaded {len(surfaces)} surfaces for {ticker} "
               f"({dates[0]} -> {dates[-1]})")

    config = BacktestConfig(
        initial_capital=capital,
        delta_hedge=delta_hedge,
        holding_days=holding_days,
        target_vega_notional=target_vega,
    )
    strategy_fn = partial(STRATEGIES[strategy], tenor=tenor)

    results = run_backtest(surfaces, strategy_fn=strategy_fn, config=config,
                           ticker=ticker)
    if results.empty:
        click.echo("Backtest produced no rows.", err=True)
        raise SystemExit(1)

    out_path = Path(output)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    m = compute_all_metrics(results)
    click.echo(f"\nResults saved to {out_path}  ({len(results)} rows)")
    click.echo(
        f"\n  {strategy} on {ticker}, tenor {tenor}y, "
        f"delta-hedge {'ON' if delta_hedge else 'OFF'}\n"
        f"  Net P&L:      {results['net_pnl'].sum():>12,.0f}\n"
        f"  Total return: {m['total_return']:>11.1%}\n"
        f"  Sharpe:       {m['sharpe']:>12.2f}\n"
        f"  Sortino:      {m['sortino']:>12.2f}\n"
        f"  Max drawdown: {m['max_drawdown']:>11.1%}\n"
        f"  Win rate:     {m['win_rate']:>11.1%}\n"
        f"  Total costs:  {results['transaction_costs'].sum():>12,.0f}"
    )
    click.echo(
        f"\n  P&L attribution\n"
        f"    delta   {results['delta_pnl'].sum():>12,.0f}\n"
        f"    gamma   {results['gamma_pnl'].sum():>12,.0f}\n"
        f"    vega    {results['vega_pnl'].sum():>12,.0f}\n"
        f"    theta   {results['theta_pnl'].sum():>12,.0f}\n"
        f"    hedge   {results['hedge_pnl'].sum():>12,.0f}\n"
        f"    residual{results['unexplained_pnl'].sum():>12,.0f}"
    )


if __name__ == "__main__":
    main()
