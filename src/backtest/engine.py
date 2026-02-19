"""
Walk-forward backtest engine for vol-arb strategies on historical surfaces.

The engine:
1. Iterates over historical surface snapshots.
2. Generates trading signals using the signal modules.
3. Executes trades through the strategy module.
4. Computes daily P&L with Greeks-based delta hedging.
5. Accumulates returns for performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents an open option position."""

    entry_date: date
    expiry_date: date
    strike: float
    option_type: str          # "call" or "put"
    position: float           # Signed notional (+ long, - short)
    entry_iv: float
    entry_price: float
    tenor_years: float
    strategy: str = ""
    delta_hedge: float = 0.0  # Current delta hedge ratio


@dataclass
class BacktestResult:
    """Container for backtest output."""

    returns: pd.Series = field(default_factory=pd.Series)
    trades: list[Trade] = field(default_factory=list)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Walk-forward backtesting engine.

    Parameters
    ----------
    transaction_cost_bps:
        Round-trip transaction cost in basis points.
    margin_rate:
        Annual financing rate for short positions.
    """

    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        margin_rate: float = 0.02,
    ) -> None:
        self.transaction_cost_bps = transaction_cost_bps
        self.margin_rate = margin_rate

    def run(
        self,
        surfaces: list,
        dates: list[date],
        strategy_fn: "callable",  # type: ignore[type-arg]
        signal_fn: Optional["callable"] = None,  # type: ignore[type-arg]
    ) -> BacktestResult:
        """Run the walk-forward backtest.

        Parameters
        ----------
        surfaces:
            Ordered list of VolSurface objects.
        dates:
            Corresponding trade dates.
        strategy_fn:
            Callable(surface, signal) → list[Trade].
        signal_fn:
            Callable(surface) → dict of signals.

        Returns
        -------
        BacktestResult
        """
        all_returns: list[float] = []
        all_trades: list[Trade] = []
        open_trades: list[Trade] = []

        for i, (surf, dt) in enumerate(zip(surfaces, dates)):
            # Generate signals
            signals: dict = {}
            if signal_fn is not None:
                try:
                    signals = signal_fn(surf)
                except Exception as exc:
                    logger.warning("Signal generation failed on %s: %s", dt, exc)

            # Close expiring trades and compute P&L
            day_pnl = 0.0
            still_open: list[Trade] = []
            for trade in open_trades:
                if trade.expiry_date <= dt:
                    pnl = self._close_trade(trade, surf)
                    day_pnl += pnl
                else:
                    # Daily mark-to-market P&L via vega * Δσ
                    mtm = self._mark_to_market(trade, surf)
                    day_pnl += mtm
                    still_open.append(trade)
            open_trades = still_open

            # Open new trades
            try:
                new_trades = strategy_fn(surf, signals)
                for trade in new_trades:
                    cost = abs(trade.entry_price) * self.transaction_cost_bps / 10000
                    day_pnl -= cost
                open_trades.extend(new_trades)
                all_trades.extend(new_trades)
            except Exception as exc:
                logger.warning("Strategy execution failed on %s: %s", dt, exc)

            all_returns.append(day_pnl)

        returns_series = pd.Series(all_returns, index=dates, name="strategy_pnl")
        result = BacktestResult(returns=returns_series, trades=all_trades)
        return result

    def _close_trade(self, trade: Trade, surface: object) -> float:
        """Compute P&L when closing / expiring a trade."""
        # TODO: Implement proper pricing at expiry
        return 0.0

    def _mark_to_market(self, trade: Trade, surface: object) -> float:
        """Daily MTM P&L approximation via vega * vol change."""
        # TODO: Implement Greeks-based daily P&L
        return 0.0
