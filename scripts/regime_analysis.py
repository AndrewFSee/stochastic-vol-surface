#!/usr/bin/env python
"""CLI: backtest every strategy across the corpus and break results out by vol regime.

This is the question the live sample could not answer: does a strategy survive
a crisis, or does it only work in calm markets?  Results are grouped by the VIX
regime thresholds in ``config/default.yaml``.

Usage:
    python scripts/regime_analysis.py --surfaces-dir data/historical/surfaces
    python scripts/regime_analysis.py --surfaces-dir data/surfaces --output report.csv
"""

import logging
import sys
from functools import partial

import click
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
for _q in ("src.surface.surface", "src.surface.batch", "src.data.storage"):
    logging.getLogger(_q).setLevel(logging.ERROR)


def _vix_regimes(start, end, cfg) -> pd.Series:
    """Daily VIX regime labels over [start, end]."""
    import yfinance as yf

    hist = yf.download("^VIX", start=start, end=end, progress=False)
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    lo = cfg["regime_low_vix"]
    hi = cfg["regime_high_vix"]
    crisis = cfg["regime_crisis_vix"]
    labels = pd.cut(
        close, [-np.inf, lo, hi, crisis, np.inf],
        labels=["Low", "Normal", "High", "Crisis"],
    )
    out = pd.Series(labels.values, index=close.index.date, name="regime")
    out.index.name = "date"
    return out


@click.command()
@click.option("--ticker", "-t", default="SPY", show_default=True)
@click.option("--surfaces-dir", default="data/historical/surfaces", show_default=True)
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--tenor", default=0.25, show_default=True)
@click.option("--holding-days", default=5, show_default=True)
@click.option("--target-vega", default=10_000.0, show_default=True)
@click.option("--capital", default=1_000_000.0, show_default=True)
@click.option("--output", default=None, help="Write the per-regime table to CSV")
def main(ticker, surfaces_dir, start, end, tenor, holding_days,
         target_vega, capital, output):
    """Backtest all strategies and summarise performance by vol regime."""
    import yaml

    from src.backtest.engine import BacktestConfig, run_backtest
    from src.backtest.metrics import compute_all_metrics
    from src.backtest.strategies import STRATEGIES
    from src.surface.batch import load_surfaces

    cfg_all = yaml.safe_load(open("config/default.yaml"))
    sig_cfg = cfg_all["signals"]

    click.echo(f"Loading surfaces for {ticker} from {surfaces_dir} …")
    surfaces = load_surfaces(ticker, surfaces_dir=surfaces_dir, start=start, end=end)
    if not surfaces:
        click.echo("No surfaces found.", err=True)
        raise SystemExit(1)

    dates = sorted(surfaces)
    click.echo(f"  {len(surfaces)} surfaces, {dates[0]} -> {dates[-1]}\n")

    regimes = _vix_regimes(
        str(dates[0]), str(dates[-1] + pd.Timedelta(days=1)), sig_cfg,
    )

    config = BacktestConfig(
        initial_capital=capital,
        holding_days=holding_days,
        target_vega_notional=target_vega,
    )

    overall_rows = []
    regime_rows = []

    for name, fn in STRATEGIES.items():
        res = run_backtest(surfaces, strategy_fn=partial(fn, tenor=tenor),
                           config=config, ticker=ticker)
        if res.empty:
            continue

        m = compute_all_metrics(res)
        overall_rows.append({
            "strategy": name,
            "total_return": m["total_return"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "win_rate": m["win_rate"],
            "net_pnl": m["total_pnl"],
            "costs": res["transaction_costs"].sum(),
        })

        res = res.copy()
        res["regime"] = res["date"].map(regimes)
        for reg, grp in res.groupby("regime", observed=True):
            if grp.empty:
                continue
            regime_rows.append({
                "strategy": name,
                "regime": reg,
                "days": len(grp),
                "net_pnl": grp["net_pnl"].sum(),
                "pnl_per_day": grp["net_pnl"].mean(),
                "hit_rate": (grp["net_pnl"] > 0).mean(),
                "gamma": grp["gamma_pnl"].sum(),
                "vega": grp["vega_pnl"].sum(),
                "theta": grp["theta_pnl"].sum(),
            })

    overall = pd.DataFrame(overall_rows)
    by_regime = pd.DataFrame(regime_rows)

    click.echo("=" * 78)
    click.echo(f"OVERALL  ({dates[0]} -> {dates[-1]}, {len(surfaces)} sessions)")
    click.echo("=" * 78)
    disp = overall.copy()
    for col in ("total_return", "max_dd", "win_rate"):
        disp[col] = (100 * disp[col]).round(1).astype(str) + "%"
    disp["sharpe"] = disp["sharpe"].round(2)
    for col in ("net_pnl", "costs"):
        disp[col] = disp[col].round(0).map("{:,.0f}".format)
    click.echo(disp.to_string(index=False))

    if not by_regime.empty:
        order = ["Low", "Normal", "High", "Crisis"]
        click.echo("\n" + "=" * 78)
        click.echo("BY VOLATILITY REGIME  (P&L per day)")
        click.echo("=" * 78)
        pivot = by_regime.pivot_table(
            index="strategy", columns="regime", values="pnl_per_day",
            observed=True,
        )
        pivot = pivot[[c for c in order if c in pivot.columns]]
        click.echo(pivot.round(0).to_string())

        click.echo("\nSession counts per regime:")
        counts = by_regime[by_regime.strategy == by_regime.strategy.iloc[0]]
        counts = counts.set_index("regime")["days"].reindex(
            [c for c in order if c in set(by_regime.regime)]
        )
        click.echo(counts.to_string())

        click.echo("\nHit rate by regime:")
        hp = by_regime.pivot_table(index="strategy", columns="regime",
                                   values="hit_rate", observed=True)
        hp = hp[[c for c in order if c in hp.columns]]
        click.echo((100 * hp).round(1).to_string())

    if output:
        from pathlib import Path

        out = Path(output)
        if str(out.parent) not in ("", "."):
            out.parent.mkdir(parents=True, exist_ok=True)
        by_regime.to_csv(out, index=False)
        overall.to_csv(out.with_name(out.stem + "_overall.csv"), index=False)
        click.echo(f"\nWritten to {out} and {out.with_name(out.stem + '_overall.csv')}")


if __name__ == "__main__":
    main()
