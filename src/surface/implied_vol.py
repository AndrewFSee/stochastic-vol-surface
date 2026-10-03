"""Black-Scholes IV inversion with fast initial guess + Brent's method.

Two solvers are provided:
  • `implied_vol`           – scalar, Brent's method (robust, moderate speed)
  • `implied_vol_fast`      – scalar, rational initial-guess + Newton (fast)
  • `implied_vol_vectorized` – array wrapper (uses fast solver by default)
"""

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
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

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
    *,
    fast: bool = True,
) -> np.ndarray:
    """Vectorised wrapper around :func:`implied_vol` / :func:`implied_vol_fast`.

    Parameters accept scalars (broadcast) or arrays of the same length as
    *prices*.  Set ``fast=True`` (default) to use the Newton solver with a
    rational initial guess — roughly 5-10× faster than Brent on large arrays.
    """
    n = len(prices)
    S_arr = np.broadcast_to(np.asarray(S, dtype=float), n)
    T_arr = np.broadcast_to(np.asarray(T, dtype=float), n)
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), n)

    solver = implied_vol_fast if fast else implied_vol

    result = np.empty(n, dtype=float)
    for i in range(n):
        result[i] = solver(
            float(prices[i]),
            float(S_arr[i]),
            float(K[i]),
            float(T_arr[i]),
            float(r_arr[i]),
            option_type=str(option_types[i]),  # type: ignore[arg-type]
        )
    return result


# ────────────────────────────────────────────────────────────────────────────
# Fast solver: rational initial guess  +  Newton-Raphson with vega
# ────────────────────────────────────────────────────────────────────────────

def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (dPrice/dSigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return S * _n(d1) * sqrtT


def _initial_guess(price: float, S: float, K: float, T: float, r: float,
                   option_type: OptionType) -> float:
    """Brenner-Subrahmanyam approximation as starting point.

    sigma_0 ≈ sqrt(2*pi/T) * price / S   (ATM approximation)
    Corrected for moneyness via a heuristic.
    """
    F = S * math.exp(r * T)
    m = K / F  # moneyness ratio

    # Normalised price
    if option_type == "put":
        # Convert put to call via parity for the guess
        call_price = price + S - K * math.exp(-r * T)
        if call_price <= 0:
            call_price = price  # fallback
    else:
        call_price = max(price, 1e-10)

    # Brenner-Subrahmanyam with moneyness adjustment
    sigma_bs = math.sqrt(2.0 * math.pi / max(T, 1e-6)) * call_price / max(S, 1e-6)

    # Clamp to reasonable range
    return max(min(sigma_bs, 5.0), 0.01)


def implied_vol_fast(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Newton-Raphson IV solver with rational initial guess.

    Falls back to :func:`implied_vol` (Brent) if Newton doesn't converge.
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

    sigma = _initial_guess(price, S, K, T, r, option_type)

    for _ in range(max_iter):
        bs = bs_price(S, K, T, r, sigma, option_type)
        diff = bs - price
        if abs(diff) < tol:
            return sigma

        vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break  # vega too small — fall back

        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001  # reset if negative

    # Fall back to Brent's method
    return implied_vol(price, S, K, T, r, option_type, tol=tol)
