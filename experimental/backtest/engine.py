"""Walk-forward backtest engine with daily mark-to-market and delta hedging.

Design
------
The unit of trading is an :class:`OptionLeg` that carries its full contract
specification — type, strike, expiry, signed quantity.  That is what makes an
honest backtest possible: every open leg can be repriced against a later
surface by looking up its *own* moneyness and *remaining* tenor, rather than
approximating P&L from a single ATM vol change.

Each day the engine:

1. Reprices every open leg on the new surface and spot (settling expired legs
   at intrinsic value).
2. Marks the delta hedge against the spot move.
3. Rebalances the hedge to the new net delta, charging costs on shares traded.
4. Closes positions that have reached their holding period or expired.
5. Opens a new position when none is active.

P&L is *realised from repricing*, not from a Greek approximation.  The Greek
columns (delta/gamma/vega/theta) are a separate attribution of that P&L, so
the residual between them and the total is a diagnostic of how much of the
move the second-order expansion fails to explain.

Sign conventions
----------------
``quantity`` is signed: positive is long, negative is short.  ``hedge_shares``
is signed in shares of the underlying.  All option quantities are multiplied
by ``contract_multiplier`` (default 100) to reach cash terms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Backtest execution parameters."""

    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    rebalance_frequency: str = "daily"

    contract_multiplier: float = 100.0
    delta_hedge: bool = True
    #: Rebalance the hedge only when net delta drifts by more than this many
    #: shares-equivalent; 0 rebalances every day.
    hedge_tolerance_shares: float = 0.0
    #: Trading days to hold a position before rolling into a fresh one.
    holding_days: int = 5
    #: Close a position once its shortest leg has less than this many days left.
    min_days_to_expiry: int = 2
    #: Target vega exposure per position, in cash per vol point (0.01).
    target_vega_notional: float = 10_000.0


# ────────────────────────────────────────────────────────────────────────────
# Instruments
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class OptionLeg:
    """One option contract with everything needed to reprice it later."""

    option_type: str            # "call" | "put"
    strike: float
    expiry: date
    quantity: float             # signed, in contracts

    entry_iv: float = float("nan")
    entry_price: float = float("nan")
    last_price: float = float("nan")
    last_iv: float = float("nan")

    def years_to_expiry(self, as_of: date) -> float:
        """Remaining tenor in years (ACT/365), floored at zero."""
        return max((self.expiry - as_of).days / DAYS_PER_YEAR, 0.0)


@dataclass
class Position:
    """A strategy instance: a bundle of legs opened on one date."""

    open_date: date
    ticker: str
    strategy: str
    legs: list[OptionLeg] = field(default_factory=list)
    entry_spot: float = float("nan")
    hedge_shares: float = 0.0
    days_held: int = 0

    def is_expired(self, as_of: date, min_days: int = 0) -> bool:
        """True once the shortest-dated leg is within *min_days* of expiry."""
        return any((leg.expiry - as_of).days <= min_days for leg in self.legs)


# Retained for backward compatibility with the previous engine API.
@dataclass
class Trade:
    """Legacy flat trade record (kept so older callers keep importing)."""

    date: date
    ticker: str
    strategy: str
    position: float
    entry_price: float
    delta: float = 0.0
    vega: float = 0.0


@dataclass
class DailyPnL:
    """One day of portfolio accounting."""

    date: date
    gross_pnl: float
    transaction_costs: float
    net_pnl: float
    portfolio_value: float
    option_pnl: float = 0.0
    hedge_pnl: float = 0.0
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    unexplained_pnl: float = 0.0
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    n_positions: int = 0
    spot: float = float("nan")


# ────────────────────────────────────────────────────────────────────────────
# Pricing helpers
# ────────────────────────────────────────────────────────────────────────────


def leg_iv(surface, leg: OptionLeg, as_of: date, spot: float, r: float) -> float:
    """Look up a leg's implied vol on *surface* at its own moneyness/tenor.

    The surface is indexed by log-forward-moneyness ``k = ln(K/F)`` with
    ``F = S·e^{rT}``, the carry forward; the surfaces now use parity forwards (see README).
    """
    T = leg.years_to_expiry(as_of)
    if T <= 0:
        return float(leg.last_iv if np.isfinite(leg.last_iv) else 0.0)
    fwd = spot * np.exp(r * T)
    k = float(np.log(leg.strike / fwd))
    iv = float(surface.iv(k, T))
    if not np.isfinite(iv) or iv <= 0:
        # Fall back to the last good mark rather than emitting a bad price.
        return float(leg.last_iv if np.isfinite(leg.last_iv) else 0.0)
    return iv


def price_leg(leg: OptionLeg, as_of: date, spot: float, r: float, iv: float) -> float:
    """Price one contract (per share, not per contract-multiplier)."""
    from experimental.backtest.greeks import bs_price

    T = leg.years_to_expiry(as_of)
    return bs_price(spot, leg.strike, T, r, iv, leg.option_type)


def leg_greeks(leg: OptionLeg, as_of: date, spot: float, r: float, iv: float) -> dict:
    """Per-share Greeks for one contract."""
    from experimental.backtest.greeks import bs_delta, bs_gamma, bs_theta, bs_vega

    T = leg.years_to_expiry(as_of)
    return {
        "delta": bs_delta(spot, leg.strike, T, r, iv, leg.option_type),
        "gamma": bs_gamma(spot, leg.strike, T, r, iv),
        "vega": bs_vega(spot, leg.strike, T, r, iv),
        "theta": bs_theta(spot, leg.strike, T, r, iv, leg.option_type),
    }


def portfolio_greeks(
    positions: list[Position], as_of: date, spot: float, r: float,
    surface, multiplier: float,
) -> dict:
    """Aggregate cash Greeks across all open positions."""
    tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for pos in positions:
        for leg in pos.legs:
            iv = leg_iv(surface, leg, as_of, spot, r)
            g = leg_greeks(leg, as_of, spot, r, iv)
            q = leg.quantity * multiplier
            for key in tot:
                tot[key] += g[key] * q
        tot["delta"] += pos.hedge_shares
    return tot


# ────────────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────────────


def run_backtest(
    surface_history: dict,
    strategy_fn: Callable,
    config: Optional[BacktestConfig] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ticker: str = "SPY",
) -> pd.DataFrame:
    """Walk-forward backtest over a dated surface history.

    Parameters
    ----------
    surface_history : dict[date, VolSurface]
    strategy_fn : callable(as_of, surface, ticker=..., config=...) -> Position | None
        Returns a new position to open, or None to stay flat.
    config : BacktestConfig
    start_date, end_date : optional inclusive date filters
    ticker : str

    Returns
    -------
    pd.DataFrame with one row per trading day.  Always contains the columns the
    tearsheet expects (``date``, ``gross_pnl``, ``net_pnl``, ``portfolio_value``)
    plus the P&L decomposition and running Greeks.
    """
    config = config or BacktestConfig()
    dates = sorted(surface_history)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if not dates:
        return pd.DataFrame()

    mult = config.contract_multiplier
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

    portfolio_value = config.initial_capital
    open_positions: list[Position] = []
    records: list[DailyPnL] = []

    prev_spot: Optional[float] = None
    prev_date: Optional[date] = None
    prev_greeks: Optional[dict] = None

    for i, dt in enumerate(dates):
        surface = surface_history[dt]
        spot = float(surface.spot)
        r = float(getattr(surface, "r", 0.05))

        option_pnl = 0.0
        hedge_pnl = 0.0
        costs = 0.0
        vega_pnl = 0.0

        # ── 1. Mark open legs to the new surface ──────────────────────────
        for pos in open_positions:
            for leg in pos.legs:
                iv_now = leg_iv(surface, leg, dt, spot, r)
                px_now = price_leg(leg, dt, spot, r, iv_now)
                if np.isfinite(leg.last_price):
                    option_pnl += (px_now - leg.last_price) * leg.quantity * mult
                    # Vega attribution uses each leg's own vol change.
                    if np.isfinite(leg.last_iv):
                        g_prev = leg_greeks(
                            leg, prev_date or dt, prev_spot or spot, r, leg.last_iv,
                        )
                        vega_pnl += (
                            g_prev["vega"] * (iv_now - leg.last_iv)
                            * leg.quantity * mult
                        )
                leg.last_price = px_now
                leg.last_iv = iv_now

        # ── 2. Mark the delta hedge against the spot move ─────────────────
        if prev_spot is not None:
            d_spot = spot - prev_spot
            for pos in open_positions:
                hedge_pnl += pos.hedge_shares * d_spot

        gross_pnl = option_pnl + hedge_pnl

        # ── 3. Greek attribution of the day's move ────────────────────────
        delta_pnl = gamma_pnl = theta_pnl = 0.0
        if prev_greeks is not None and prev_spot is not None and prev_date is not None:
            d_spot = spot - prev_spot
            dt_days = (dt - prev_date).days
            # Option delta only; the hedge is accounted separately above.
            delta_pnl = prev_greeks["delta_options"] * d_spot
            gamma_pnl = 0.5 * prev_greeks["gamma"] * d_spot ** 2
            theta_pnl = prev_greeks["theta"] * dt_days
        unexplained = gross_pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl + hedge_pnl)

        # ── 4. Close positions that have aged out or are near expiry ──────
        still_open: list[Position] = []
        for pos in open_positions:
            pos.days_held += 1
            expiring = pos.is_expired(dt, config.min_days_to_expiry)
            if pos.days_held >= config.holding_days or expiring:
                costs += _closing_cost(pos, spot, mult, cost_rate)
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 5. Open a new position when flat ──────────────────────────────
        if not open_positions:
            new_pos = _open_position(strategy_fn, dt, surface, ticker, config, spot, r)
            if new_pos is not None:
                open_positions.append(new_pos)
                costs += _opening_cost(new_pos, spot, mult, cost_rate)

        # ── 6. Rebalance the delta hedge ──────────────────────────────────
        greeks_now = portfolio_greeks(open_positions, dt, spot, r, surface, mult)
        option_delta = greeks_now["delta"] - sum(p.hedge_shares for p in open_positions)

        if config.delta_hedge and open_positions:
            target_total = -option_delta
            current_total = sum(p.hedge_shares for p in open_positions)
            adjustment = target_total - current_total
            if abs(adjustment) > config.hedge_tolerance_shares:
                costs += abs(adjustment) * spot * cost_rate
                # Attribute the whole hedge to the first position for simplicity;
                # the portfolio-level net is what matters for P&L.
                open_positions[0].hedge_shares += adjustment

        net_pnl = gross_pnl - costs
        portfolio_value += net_pnl

        final_greeks = portfolio_greeks(open_positions, dt, spot, r, surface, mult)
        records.append(DailyPnL(
            date=dt,
            gross_pnl=gross_pnl,
            transaction_costs=costs,
            net_pnl=net_pnl,
            portfolio_value=portfolio_value,
            option_pnl=option_pnl,
            hedge_pnl=hedge_pnl,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            theta_pnl=theta_pnl,
            unexplained_pnl=unexplained,
            net_delta=final_greeks["delta"],
            net_gamma=final_greeks["gamma"],
            net_vega=final_greeks["vega"],
            net_theta=final_greeks["theta"],
            n_positions=len(open_positions),
            spot=spot,
        ))

        prev_spot = spot
        prev_date = dt
        prev_greeks = {
            "delta_options": final_greeks["delta"] - sum(
                p.hedge_shares for p in open_positions
            ),
            "gamma": final_greeks["gamma"],
            "vega": final_greeks["vega"],
            "theta": final_greeks["theta"],
        }

    df = pd.DataFrame([r.__dict__ for r in records])
    if not df.empty:
        df["cumulative_pnl"] = df["net_pnl"].cumsum()
    return df


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _open_position(strategy_fn, dt, surface, ticker, config, spot, r) -> Optional[Position]:
    """Call the strategy and initialise the new position's marks."""
    try:
        pos = strategy_fn(dt, surface, ticker=ticker, config=config)
    except TypeError:
        # Tolerate simpler strategy signatures.
        pos = strategy_fn(dt, surface)
    except Exception as exc:
        logger.warning("Strategy failed on %s: %s", dt, exc)
        return None

    if pos is None or not getattr(pos, "legs", None):
        return None

    for leg in pos.legs:
        iv = leg_iv(surface, leg, dt, spot, r)
        px = price_leg(leg, dt, spot, r, iv)
        leg.entry_iv = iv
        leg.entry_price = px
        leg.last_iv = iv
        leg.last_price = px
    pos.entry_spot = spot
    return pos


def _opening_cost(pos: Position, spot: float, mult: float, rate: float) -> float:
    """Transaction cost of establishing a position's option legs."""
    return sum(
        abs(leg.quantity) * mult * max(leg.entry_price, 0.0) * rate
        for leg in pos.legs
    )


def _closing_cost(pos: Position, spot: float, mult: float, rate: float) -> float:
    """Transaction cost of unwinding a position, including its stock hedge."""
    opt = sum(
        abs(leg.quantity) * mult * max(leg.last_price, 0.0) * rate
        for leg in pos.legs
    )
    return opt + abs(pos.hedge_shares) * spot * rate
