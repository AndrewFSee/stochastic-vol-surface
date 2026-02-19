"""
Greeks calculation for options P&L attribution.

Computes delta, gamma, vega, theta (and rho) using the Black-Scholes
closed-form formulas.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from scipy.stats import norm


class Greeks(NamedTuple):
    """Container for option Greeks."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """Compute Black-Scholes Greeks analytically.

    Parameters
    ----------
    S:
        Spot price.
    K:
        Strike price.
    T:
        Time to expiry in years.
    r:
        Continuously-compounded risk-free rate.
    sigma:
        Implied volatility.
    option_type:
        ``"call"`` or ``"put"``.

    Returns
    -------
    Greeks
        Named tuple of delta, gamma, vega, theta, rho.
    """
    if T <= 0 or sigma <= 0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_nd1 = norm.cdf(-d1)
    N_nd2 = norm.cdf(-d2)

    discount = math.exp(-r * T)

    gamma = n_d1 / (S * sigma * math.sqrt(T))
    vega = S * n_d1 * math.sqrt(T) / 100  # Per 1% vol move

    if option_type == "call":
        delta = N_d1
        theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) - r * K * discount * N_d2) / 365
        rho = K * T * discount * N_d2 / 100  # Per 1% rate move
    else:
        delta = N_d1 - 1
        theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) + r * K * discount * N_nd2) / 365
        rho = -K * T * discount * N_nd2 / 100

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
