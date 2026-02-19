"""
SABR stochastic volatility model.

Implements the Hagan et al. (2002) lognormal approximation formula for the
implied volatility smile.  Calibrates (α, ρ, ν) for a fixed backbone β.

Reference:
    Hagan, P.S., Kumar, D., Lesniewski, A.S., Woodward, D.E. (2002).
    Managing smile risk. Wilmott Magazine, 1, 84-108.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute SABR implied volatility using the Hagan (2002) approximation.

    Parameters
    ----------
    F:
        Forward price.
    K:
        Strike price.
    T:
        Time to expiry in years.
    alpha:
        Initial volatility (α).
    beta:
        Backbone exponent (β ∈ [0, 1]; 0=normal, 1=log-normal).
    rho:
        Correlation between spot and vol (ρ ∈ (-1, 1)).
    nu:
        Volatility-of-volatility (ν > 0).

    Returns
    -------
    float
        SABR implied Black-Scholes volatility.
    """
    if T <= 0 or alpha <= 0 or nu <= 0:
        return float("nan")

    eps = 1e-10
    if abs(F - K) < eps:  # ATM formula
        FK_mid = F
        denom = (FK_mid ** (1 - beta)) * (
            1
            + (((1 - beta) ** 2 / 24) * (alpha**2 / FK_mid ** (2 * (1 - beta)))
               + 0.25 * rho * beta * nu * alpha / FK_mid ** (1 - beta)
               + (2 - 3 * rho**2) / 24 * nu**2) * T
        )
        return alpha / denom

    FK = F * K
    FK_beta = FK ** ((1 - beta) / 2)

    log_FK = np.log(F / K)

    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log(
        (np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
    )

    if abs(x_z) < eps:
        zz = 1.0
    else:
        zz = z / x_z

    A = alpha / (
        FK_beta
        * (
            1
            + ((1 - beta) ** 2 / 24) * log_FK**2
            + ((1 - beta) ** 4 / 1920) * log_FK**4
        )
    )

    B = 1 + (
        ((1 - beta) ** 2 / 24) * alpha**2 / FK ** (1 - beta)
        + (rho * beta * nu * alpha) / (4 * FK ** ((1 - beta) / 2))
        + ((2 - 3 * rho**2) / 24) * nu**2
    ) * T

    return A * zz * B


def sabr_vol_smile(
    F: float,
    strikes: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Vectorized SABR vol smile over an array of strikes."""
    return np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])


def calibrate_sabr(
    F: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    beta: float = 0.5,
    alpha_0: float = 0.30,
    rho_0: float = -0.30,
    nu_0: float = 0.40,
) -> dict[str, float]:
    """Calibrate SABR (α, ρ, ν) to market implied vols.

    Parameters
    ----------
    F:
        Forward price.
    strikes:
        Array of observed strike prices.
    market_vols:
        Corresponding market implied vols (decimal).
    T:
        Time to expiry in years.
    beta:
        Fixed backbone exponent.
    alpha_0, rho_0, nu_0:
        Initial parameter guesses.

    Returns
    -------
    dict[str, float]
        Calibrated parameters: ``{alpha, beta, rho, nu, rmse}``.
    """
    valid = np.isfinite(market_vols) & np.isfinite(strikes)
    strikes = strikes[valid]
    market_vols = market_vols[valid]

    if len(strikes) < 3:
        logger.warning("Too few data points for SABR calibration")
        return {"alpha": alpha_0, "beta": beta, "rho": rho_0, "nu": nu_0, "rmse": float("nan")}

    def _objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        model_vols = sabr_vol_smile(F, strikes, T, alpha, beta, rho, nu)
        residuals = model_vols - market_vols
        return float(np.mean(residuals**2))

    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]
    result = minimize(
        _objective,
        x0=[alpha_0, rho_0, nu_0],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    alpha_opt, rho_opt, nu_opt = result.x
    rmse = np.sqrt(result.fun)

    logger.debug(
        "SABR calibrated: α=%.4f ρ=%.4f ν=%.4f RMSE=%.6f",
        alpha_opt, rho_opt, nu_opt, rmse,
    )
    return {
        "alpha": float(alpha_opt),
        "beta": float(beta),
        "rho": float(rho_opt),
        "nu": float(nu_opt),
        "rmse": float(rmse),
    }
