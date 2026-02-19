"""
SVI (Stochastic Volatility Inspired) parameterization (Gatheral 2006).

The raw SVI parameterization expresses total implied variance as:

    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))

where k = log(K/F) is log-moneyness and w = σ_IV² * T is total variance.

This quasi-explicit form admits a fast calibration via Nelder-Mead or
L-BFGS-B with good initial guesses derived from the slice statistics.

Reference:
    Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide.
    Wiley Finance.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Compute SVI total variance w(k) = a + b*(ρ*(k-m) + sqrt((k-m)²+σ²)).

    Parameters
    ----------
    k:
        Log-moneyness array.
    a, b, rho, m, sigma:
        SVI parameters:
        - a: overall variance level (shifts the smile up/down)
        - b: slope / wing weight  (b > 0)
        - rho: skew (rho ∈ (-1, 1))
        - m: horizontal shift (smile location)
        - sigma: smile curvature / ATM convexity (sigma > 0)

    Returns
    -------
    np.ndarray
        Total variance w(k) ≥ 0 for all k.
    """
    km = k - m
    return a + b * (rho * km + np.sqrt(km**2 + sigma**2))


def svi_smile(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    T: float = 1.0,
) -> np.ndarray:
    """Convert SVI total variance to implied vol.

    Parameters
    ----------
    k:
        Log-moneyness array.
    T:
        Tenor in years (divides total variance to get σ²).

    Returns
    -------
    np.ndarray
        Implied volatility array.
    """
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 0.0)
    if T <= 0:
        return np.full_like(k, float("nan"))
    return np.sqrt(w / T)


def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float = 1.0,
    max_iter: int = 1000,
) -> Optional[tuple[float, float, float, float, float]]:
    """Fit the SVI parameterization to a single tenor slice.

    Parameters
    ----------
    k:
        Log-moneyness values.
    iv:
        Market implied vols.
    T:
        Tenor in years (used to convert IV to total variance).
    max_iter:
        Maximum L-BFGS-B iterations.

    Returns
    -------
    tuple (a, b, rho, m, sigma) | None
        Fitted SVI parameters or ``None`` if fitting fails.
    """
    valid = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k = k[valid]
    iv = iv[valid]

    if len(k) < 3:
        return None

    w_market = iv**2 * T

    # Initial guess from data statistics
    w_atm = float(np.interp(0.0, k, w_market)) if k.min() <= 0 <= k.max() else w_market.mean()
    a0 = max(w_atm * 0.9, 1e-4)
    b0 = 0.10
    rho0 = -0.30
    m0 = 0.0
    sigma0 = 0.20

    def _objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if a + b * sigma * (1 - abs(rho)) < 0:
            return 1e10  # Non-negativity of w
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        if np.any(w_model < 0):
            return 1e10
        return float(np.mean((w_model - w_market) ** 2))

    bounds = [
        (0.0, 2.0),      # a
        (1e-6, 2.0),     # b
        (-0.999, 0.999), # rho
        (-1.0, 1.0),     # m
        (1e-6, 2.0),     # sigma
    ]

    try:
        result = minimize(
            _objective,
            x0=[a0, b0, rho0, m0, sigma0],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12},
        )
        if not result.success or result.fun > 1e-2:
            # Retry with differential evolution
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
            )
            result = de_result

        a, b, rho, m, sigma = result.x
        return float(a), float(b), float(rho), float(m), float(sigma)
    except Exception as exc:
        logger.warning("SVI fit failed: %s", exc)
        return None
