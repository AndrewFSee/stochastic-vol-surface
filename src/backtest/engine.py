"""Walk-forward backtest engine with daily rebalancing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    rebalance_frequency: str = "daily"


@dataclass
class Trade:
    date: date
    ticker: str
    strategy: str
    position: float          # number of contracts
    entry_price: float
    delta: float = 0.0
    vega: float = 0.0


@dataclass
class DailyPnL:
    date: date
    gross_pnl: float
    transaction_costs: float
    net_pnl: float
    portfolio_value: float
    trades: list[Trade] = field(default_factory=list)


def run_backtest(
    surface_history: dict[date, "VolSurface"],  # noqa: F821
    strategy_fn: Callable,
    config: BacktestConfig | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest engine.

    Parameters
    ----------
    surface_history : dict mapping date → VolSurface
    strategy_fn     : callable(date, surface) → list[Trade]
    config          : BacktestConfig
    start_date, end_date : optional date range filter

    Returns
    -------
    pd.DataFrame with columns: date, gross_pnl, transaction_costs, net_pnl,
                                portfolio_value, cumulative_pnl
    """
    config = config or BacktestConfig()
    sorted_dates = sorted(surface_history.keys())

    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
    if end_date:
        sorted_dates = [d for d in sorted_dates if d <= end_date]

    portfolio_value = config.initial_capital
    records: list[DailyPnL] = []
    open_trades: list[Trade] = []

    for i, dt in enumerate(sorted_dates):
        surface = surface_history[dt]

        # Generate new trades from strategy
        new_trades = strategy_fn(dt, surface)
        total_costs = 0.0

        for trade in new_trades:
            cost = abs(trade.position * trade.entry_price) * (
                config.transaction_cost_bps + config.slippage_bps
            ) / 10_000
            total_costs += cost
            open_trades.append(trade)

        # Mark-to-market open positions (simplified: P&L from vega * IV change)
        gross_pnl = 0.0
        if i > 0 and open_trades:
            prev_date = sorted_dates[i - 1]
            prev_surface = surface_history[prev_date]
            for trade in open_trades:
                iv_change = surface.atm_vol(0.25) - prev_surface.atm_vol(0.25)
                gross_pnl += trade.vega * iv_change * trade.position

        net_pnl = gross_pnl - total_costs
        portfolio_value += net_pnl

        records.append(
            DailyPnL(
                date=dt,
                gross_pnl=gross_pnl,
                transaction_costs=total_costs,
                net_pnl=net_pnl,
                portfolio_value=portfolio_value,
                trades=new_trades,
            )
        )

        # Simple risk management: close all trades daily (delta-hedge)
        open_trades.clear()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "gross_pnl": r.gross_pnl,
                "transaction_costs": r.transaction_costs,
                "net_pnl": r.net_pnl,
                "portfolio_value": r.portfolio_value,
            }
            for r in records
        ]
    )
    if not df.empty:
        df["cumulative_pnl"] = df["net_pnl"].cumsum()
    return df
