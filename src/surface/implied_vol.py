"""Black-Scholes IV inversion using Brent's method."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


OptionType = Literal["call", "put"]

_N = norm.cdf
_n = norm.pdf


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes price for a European option.

    Parameters
    ----------
    S : float  Spot price
    K : float  Strike
    T : float  Time to expiry in years
    r : float  Continuously-compounded risk-free rate
    sigma : float  Implied volatility (annualised)
    option_type : 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        # Intrinsic value
        if option_type == "call":
            return max(S * math.exp(-r * 0) - K * math.exp(-r * 0), 0.0)
        else:
            return max(K * math.exp(-r * 0) - S * math.exp(-r * 0), 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        price = S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    else:
        price = K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)

    return float(price)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> float:
    """Invert Black-Scholes to find implied volatility using Brent's method.

    Returns NaN if the solution cannot be found (e.g., price outside
    arbitrage bounds).
    """
    if T <= 0 or price <= 0:
        return float("nan")

    # Lower bound: intrinsic value check
    intrinsic = (
        max(S - K * math.exp(-r * T), 0.0)
        if option_type == "call"
        else max(K * math.exp(-r * T) - S, 0.0)
    )
    if price < intrinsic - tol:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - price

    sigma_lo, sigma_hi = 1e-6, 20.0

    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if f_lo * f_hi > 0:
            return float("nan")
        sol = brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=max_iter)
        return float(sol)
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol_vectorized(
    prices: np.ndarray,
    S: float | np.ndarray,
    K: np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    option_types: list[OptionType] | np.ndarray,
) -> np.ndarray:
    """Vectorised wrapper around :func:`implied_vol`.

    Parameters accept scalars (broadcast) or arrays of the same length as
    *prices*.
    """
    n = len(prices)
    S_arr = np.broadcast_to(S, n)
    T_arr = np.broadcast_to(T, n)
    r_arr = np.broadcast_to(r, n)

    result = np.empty(n, dtype=float)
    for i in range(n):
        result[i] = implied_vol(
            float(prices[i]),
            float(S_arr[i]),
            float(K[i]),
            float(T_arr[i]),
            float(r_arr[i]),
            option_type=str(option_types[i]),  # type: ignore[arg-type]
        )
    return result
