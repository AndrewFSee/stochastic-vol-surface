"""
Vol-arb strategy implementations: delta-hedged straddles, risk reversals,
and calendar spreads.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np

from src.backtest.engine import Trade

logger = logging.getLogger(__name__)


def delta_hedged_straddle(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    signals: dict,
    as_of: date,
    tenor_years: float = 1 / 12,
    notional: float = 1.0,
) -> list[Trade]:
    """Open a delta-hedged ATM straddle.

    Buys an ATM call + ATM put when signals suggest long vol entry.

    Returns
    -------
    list[Trade]
        List of new trades (call + put).
    """
    action = signals.get("action", "flat")
    if action != "buy_vol":
        return []

    atm_iv = surface.get_atm_vol(tenor_years)
    if not np.isfinite(atm_iv):
        return []

    expiry = as_of + timedelta(days=int(tenor_years * 365))
    spot = surface.spot
    strike = spot  # ATM

    trades = []
    for opt_type in ("call", "put"):
        # Approximate price via Black-Scholes
        from src.surface.implied_vol import black_scholes_price

        price = black_scholes_price(spot, strike, tenor_years, 0.05, atm_iv, opt_type)
        trades.append(Trade(
            entry_date=as_of,
            expiry_date=expiry,
            strike=strike,
            option_type=opt_type,
            position=notional / 2,
            entry_iv=atm_iv,
            entry_price=price,
            tenor_years=tenor_years,
            strategy="straddle",
        ))
    return trades


def risk_reversal(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    signals: dict,
    as_of: date,
    tenor_years: float = 1 / 12,
    notional: float = 1.0,
) -> list[Trade]:
    """Open a risk reversal (sell put / buy call or vice versa) on skew signals.

    Sell skew when skew_signal = +1 (steep), buy skew when -1 (flat).

    Returns
    -------
    list[Trade]
    """
    skew_signal = signals.get("skew_signal", 0)
    if skew_signal == 0:
        return []

    spot = surface.spot
    expiry = as_of + timedelta(days=int(tenor_years * 365))

    # 25Δ strikes approximated as ±10% in log-moneyness
    k_put = -0.10
    k_call = 0.10
    strike_put = float(spot * np.exp(k_put))
    strike_call = float(spot * np.exp(k_call))

    iv_put = surface.get_iv(k_put, tenor_years)
    iv_call = surface.get_iv(k_call, tenor_years)

    from src.surface.implied_vol import black_scholes_price

    price_put = black_scholes_price(spot, strike_put, tenor_years, 0.05, iv_put, "put")
    price_call = black_scholes_price(spot, strike_call, tenor_years, 0.05, iv_call, "call")

    # Sell skew: sell put + buy call (skew_signal = +1)
    put_pos = -notional if skew_signal > 0 else notional
    call_pos = notional if skew_signal > 0 else -notional

    return [
        Trade(entry_date=as_of, expiry_date=expiry, strike=strike_put,
              option_type="put", position=put_pos, entry_iv=iv_put,
              entry_price=price_put, tenor_years=tenor_years, strategy="risk_reversal"),
        Trade(entry_date=as_of, expiry_date=expiry, strike=strike_call,
              option_type="call", position=call_pos, entry_iv=iv_call,
              entry_price=price_call, tenor_years=tenor_years, strategy="risk_reversal"),
    ]
