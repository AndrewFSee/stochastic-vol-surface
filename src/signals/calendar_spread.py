"""
Calendar spread arbitrage detection.

A calendar spread (buy long-dated vol, sell short-dated vol at the same strike)
should be non-negative in the absence of arbitrage (total variance must be
non-decreasing in tenor).

Significant calendar spread mispricings indicate either data errors or
exploitable vol structure differences across expirations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_calendar_arbitrage(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    log_moneyness_points: list[float] | None = None,
    min_spread_bps: float = 50.0,
) -> pd.DataFrame:
    """Identify calendar spread opportunities in the vol surface.

    Compares total variance w(k, T) = σ(k,T)² * T across tenor pairs.
    Flags pairs where w(k, T_short) > w(k, T_long) (calendar arbitrage).
    Also flags large positive spreads as potential calendar spread trades.

    Parameters
    ----------
    surface:
        VolSurface object.
    log_moneyness_points:
        Moneyness levels to check (default: [-0.10, 0.0, 0.10]).
    min_spread_bps:
        Minimum spread in basis points (vol bps) to flag as a signal.

    Returns
    -------
    pd.DataFrame
        Rows are (moneyness, short_tenor, long_tenor) triplets with columns:
        ``w_short``, ``w_long``, ``spread``, ``arb_flag``, ``trade_signal``.
    """
    if log_moneyness_points is None:
        log_moneyness_points = [-0.10, 0.0, 0.10]

    tenors = surface._tenors
    rows = []

    for k in log_moneyness_points:
        for i in range(len(tenors) - 1):
            T_short = tenors[i]
            T_long = tenors[i + 1]
            iv_short = surface.get_iv(k, T_short)
            iv_long = surface.get_iv(k, T_long)

            if not (np.isfinite(iv_short) and np.isfinite(iv_long)):
                continue

            w_short = iv_short**2 * T_short
            w_long = iv_long**2 * T_long
            spread = w_long - w_short
            spread_bps = spread * 10000

            arb_flag = bool(spread < 0)
            trade_signal = 1 if spread_bps > min_spread_bps else 0

            rows.append({
                "log_moneyness": k,
                "short_tenor": T_short,
                "long_tenor": T_long,
                "iv_short": iv_short,
                "iv_long": iv_long,
                "w_short": w_short,
                "w_long": w_long,
                "spread": spread,
                "spread_bps": spread_bps,
                "arb_flag": arb_flag,
                "trade_signal": trade_signal,
            })

    return pd.DataFrame(rows)
