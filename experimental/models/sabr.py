"""SABR model: Hagan (2002) implied vol formula and L-BFGS-B calibration."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import minimize


def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    eps: float = 1e-7,
) -> float:
    """Hagan (2002) SABR approximation for implied volatility.

    Parameters
    ----------
    F     : Forward price
    K     : Strike
    T     : Time-to-expiry in years
    alpha : Initial vol (α > 0)
    beta  : CEV exponent ∈ [0, 1]
    rho   : Correlation ∈ (-1, 1)
    nu    : Vol-of-vol (ν ≥ 0)
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return float("nan")

    # ATM case
    if abs(F - K) < eps:
        FK_beta = F ** (1.0 - beta)
        A = alpha / FK_beta
        B1 = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / FK_beta ** 2)
            + (rho * beta * nu * alpha / (4.0 * FK_beta))
            + ((2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
        ) * T
        return A * B1

    log_FK = math.log(F / K)
    FK_mid = math.sqrt(F * K)
    FK_mid_beta = FK_mid ** (1.0 - beta)

    z = (nu / alpha) * FK_mid_beta * log_FK
    chi_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    if abs(chi_z) < eps:
        x_chi = 1.0
    else:
        x_chi = z / chi_z

    # Leading term
    A = alpha / (
        FK_mid_beta
        * (
            1.0
            + ((1.0 - beta) ** 2 / 24.0) * log_FK ** 2
            + ((1.0 - beta) ** 4 / 1920.0) * log_FK ** 4
        )
    )

    # Correction
    B = (
        1.0
        + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / FK_mid_beta ** 2)
            + (rho * beta * nu * alpha / (4.0 * FK_mid_beta))
            + ((2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
        )
        * T
    )

    return A * x_chi * B


def calibrate_sabr(
    F: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    beta: float = 0.5,
    x0: np.ndarray | None = None,
) -> dict[str, float]:
    """Calibrate SABR parameters (alpha, rho, nu) with fixed beta using L-BFGS-B.

    Returns
    -------
    dict with keys: alpha, beta, rho, nu, rmse
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    if x0 is None:
        x0 = np.array([market_vols.mean(), -0.3, 0.5])

    def objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu < 0 or abs(rho) >= 1:
            return 1e9
        model_vols = np.array(
            [sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        )
        mask = np.isfinite(model_vols)
        if not mask.any():
            return 1e9
        return float(np.mean((model_vols[mask] - market_vols[mask]) ** 2))

    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

    alpha, rho, nu = res.x
    model_vols = np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return dict(alpha=float(alpha), beta=float(beta), rho=float(rho), nu=float(nu), rmse=rmse)
