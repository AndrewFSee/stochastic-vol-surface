"""SVI parameterization (Gatheral): raw SVI fit with quasi-explicit calibration."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution


def svi_raw(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Gatheral raw SVI total variance.

    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_natural(k: np.ndarray, delta: float, mu: float, rho: float, omega: float, zeta: float) -> np.ndarray:
    """SVI in natural parameterization."""
    return delta + omega / 2 * (1 + zeta * rho * (k - mu) + np.sqrt((zeta * (k - mu) + rho) ** 2 + 1 - rho ** 2))


def _svi_arbitrage_free(a: float, b: float, rho: float, m: float, sigma: float) -> bool:
    """Quick check: b >= 0, |rho| < 1, sigma > 0."""
    return b >= 0 and abs(rho) < 1 and sigma > 0 and a + b * sigma * np.sqrt(1 - rho ** 2) >= 0


def calibrate_svi(
    log_moneyness: np.ndarray,
    market_total_var: np.ndarray,
    x0: np.ndarray | None = None,
) -> dict[str, float]:
    """Quasi-explicit SVI calibration.

    Uses a two-step approach:
    1. Differential evolution for global search
    2. L-BFGS-B for local refinement

    Parameters
    ----------
    log_moneyness : np.ndarray  k = log(K/F)
    market_total_var : np.ndarray  w = sigma_impl^2 * T

    Returns
    -------
    dict with keys: a, b, rho, m, sigma, rmse
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(market_total_var, dtype=float)

    def loss(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        if not _svi_arbitrage_free(a, b, rho, m, sigma):
            return 1e9
        w_fit = svi_raw(k, a, b, rho, m, sigma)
        return float(np.mean((w_fit - w) ** 2))

    bounds = [
        (-0.5, max(w) * 2),       # a
        (0.0, 2.0),                # b
        (-0.999, 0.999),           # rho
        (k.min() - 0.5, k.max() + 0.5),  # m
        (1e-4, 2.0),               # sigma
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Global search
        de_res = differential_evolution(
            loss, bounds, maxiter=500, tol=1e-8, seed=42, workers=1
        )
        # Local refinement
        res = minimize(
            loss, de_res.x, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14}
        )

    a, b, rho, m, sigma = res.x
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    rmse = float(np.sqrt(np.mean((w_fit - w) ** 2)))

    return dict(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma), rmse=rmse)
