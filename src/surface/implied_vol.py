"""
Implied volatility calculation from option prices using Brent's method
applied to the Black-Scholes pricing formula.

Supports both European calls and puts.  Falls back gracefully when the
price is below intrinsic value or outside the no-arbitrage bounds.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

_SQRT_2PI = math.sqrt(2 * math.pi)
_MIN_IV = 1e-6
_MAX_IV = 10.0


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Compute the Black-Scholes option price.

    Parameters
    ----------
    S:
        Spot price of the underlying.
    K:
        Strike price.
    T:
        Time to expiry in years.
    r:
        Continuously-compounded risk-free rate.
    sigma:
        Annualised volatility (e.g. 0.20 = 20 %).
    option_type:
        ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Option price.
    """
    if T <= 0 or sigma <= 0:
        # Return intrinsic value
        if option_type == "call":
            return max(S - K * math.exp(-r * T), 0.0)
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    from scipy.stats import norm

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Optional[float]:
    """Compute implied volatility from a market option price via Brent's method.

    Inverts the Black-Scholes formula numerically.  Returns ``None`` if the
    price is outside no-arbitrage bounds or if the root-finding fails.

    Parameters
    ----------
    market_price:
        Observed mid-price of the option.
    S:
        Spot price.
    K:
        Strike price.
    T:
        Time to expiry in years.
    r:
        Continuously-compounded risk-free rate.
    option_type:
        ``"call"`` or ``"put"``.
    tol:
        Absolute tolerance for Brent's method.
    max_iter:
        Maximum iterations for the root-finder.

    Returns
    -------
    float | None
        Implied volatility in decimal form (e.g. 0.20), or ``None`` on failure.
    """
    if T <= 0 or market_price <= 0:
        return None

    # No-arbitrage bounds
    if option_type == "call":
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        upper_bound = S
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)
        upper_bound = K * math.exp(-r * T)

    if market_price < intrinsic * 0.999:
        return None  # Below intrinsic — arbitrage violated
    if market_price > upper_bound * 1.001:
        return None  # Above theoretical maximum

    def _objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Check that the objective has opposite signs at the brackets
        fa = _objective(_MIN_IV)
        fb = _objective(_MAX_IV)
        if fa * fb > 0:
            # Market price unreachable in [_MIN_IV, _MAX_IV]
            return None
        iv = brentq(_objective, _MIN_IV, _MAX_IV, xtol=tol, maxiter=max_iter)
        return float(iv)
    except ValueError:
        return None


def implied_vol_vectorized(
    market_prices: np.ndarray,
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    option_types: np.ndarray,
) -> np.ndarray:
    """Vectorized wrapper around :func:`implied_vol`.

    Parameters
    ----------
    market_prices, K, T, option_types:
        Arrays of equal length.
    S, r:
        Scalars.

    Returns
    -------
    np.ndarray
        Array of implied vols; ``np.nan`` where calculation fails.
    """
    n = len(market_prices)
    ivs = np.full(n, np.nan)
    for i in range(n):
        iv = implied_vol(
            float(market_prices[i]),
            S,
            float(K[i]),
            float(T[i]),
            r,
            str(option_types[i]),
        )
        if iv is not None:
            ivs[i] = iv
    return ivs
