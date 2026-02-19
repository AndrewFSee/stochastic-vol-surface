"""Backtest strategies: delta-hedged straddles, risk-reversals, butterfly trades."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from src.backtest.engine import Trade
from src.backtest.greeks import bs_vega, bs_delta

if TYPE_CHECKING:
    from src.surface.surface import VolSurface


def delta_hedged_straddle(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    notional: float = 100_000.0,
) -> list[Trade]:
    """Enter a delta-hedged ATM straddle.

    Buys one ATM call + one ATM put, then delta-hedges both legs.
    """
    spot = surface.spot
    atm_iv = surface.atm_vol(tenor)
    r = 0.05  # approximate risk-free rate

    vega = bs_vega(spot, spot, tenor, r, atm_iv)
    n_contracts = notional / max(vega * spot, 1.0)

    call_delta = bs_delta(spot, spot, tenor, r, atm_iv, "call")
    put_delta = bs_delta(spot, spot, tenor, r, atm_iv, "put")

    trades = [
        Trade(date=dt, ticker=ticker, strategy="straddle_call",
              position=n_contracts, entry_price=spot * atm_iv * np.sqrt(tenor / (2 * np.pi)),
              delta=call_delta * n_contracts, vega=vega * n_contracts),
        Trade(date=dt, ticker=ticker, strategy="straddle_put",
              position=n_contracts, entry_price=spot * atm_iv * np.sqrt(tenor / (2 * np.pi)),
              delta=put_delta * n_contracts, vega=vega * n_contracts),
    ]
    return trades


def risk_reversal(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    delta: float = 0.25,
    notional: float = 100_000.0,
) -> list[Trade]:
    """25-delta risk reversal: sell put, buy call."""
    spot = surface.spot
    r = 0.05
    atm_iv = surface.atm_vol(tenor)
    dk = delta * atm_iv * np.sqrt(tenor)

    call_k = spot * np.exp(dk)
    put_k = spot * np.exp(-dk)
    call_iv = float(surface.iv(dk, tenor))
    put_iv = float(surface.iv(-dk, tenor))

    vega_call = bs_vega(spot, call_k, tenor, r, call_iv)
    vega_put = bs_vega(spot, put_k, tenor, r, put_iv)
    n_contracts = notional / max(vega_call * spot, 1.0)

    trades = [
        Trade(date=dt, ticker=ticker, strategy="rr_long_call",
              position=n_contracts, entry_price=call_iv,
              delta=bs_delta(spot, call_k, tenor, r, call_iv, "call") * n_contracts,
              vega=vega_call * n_contracts),
        Trade(date=dt, ticker=ticker, strategy="rr_short_put",
              position=-n_contracts, entry_price=put_iv,
              delta=bs_delta(spot, put_k, tenor, r, put_iv, "put") * n_contracts,
              vega=vega_put * n_contracts),
    ]
    return trades


def butterfly_trade(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    wing_width: float = 0.05,
    notional: float = 100_000.0,
) -> list[Trade]:
    """ATM butterfly: buy wings, sell body."""
    spot = surface.spot
    r = 0.05
    atm_iv = surface.atm_vol(tenor)
    vega = bs_vega(spot, spot, tenor, r, atm_iv)
    n = notional / max(vega * spot, 1.0)

    lo_iv = float(surface.iv(-wing_width, tenor))
    hi_iv = float(surface.iv(wing_width, tenor))

    return [
        Trade(date=dt, ticker=ticker, strategy="bf_lo_wing",
              position=n, entry_price=lo_iv, vega=bs_vega(spot, spot * np.exp(-wing_width), tenor, r, lo_iv) * n),
        Trade(date=dt, ticker=ticker, strategy="bf_short_body",
              position=-2 * n, entry_price=atm_iv, vega=vega * 2 * n),
        Trade(date=dt, ticker=ticker, strategy="bf_hi_wing",
              position=n, entry_price=hi_iv, vega=bs_vega(spot, spot * np.exp(wing_width), tenor, r, hi_iv) * n),
    ]
