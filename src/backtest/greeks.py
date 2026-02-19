"""Black-Scholes Greeks calculator."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]
_N = norm.cdf
_n = norm.pdf


def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "call" and S > K) else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(_N(d1))
    return float(_N(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(_n(d1) / (S * sigma * math.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (per unit of volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(S * _n(d1) * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes theta (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term1 = -S * _n(d1) * sigma / (2 * math.sqrt(T))
    if option_type == "call":
        theta = term1 - r * K * math.exp(-r * T) * _N(d2)
    else:
        theta = term1 + r * K * math.exp(-r * T) * _N(-d2)
    return float(theta / 365.25)


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes rho."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(K * T * math.exp(-r * T) * _N(d2))
    return float(-K * T * math.exp(-r * T) * _N(-d2))


def bs_vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vanna: d(delta)/d(sigma) = d(vega)/dS."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return float(-_n(d1) * d2 / sigma)


def bs_volga(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes volga (vomma): d^2(price)/d(sigma)^2."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return float(S * _n(d1) * math.sqrt(T) * d1 * d2 / sigma)
