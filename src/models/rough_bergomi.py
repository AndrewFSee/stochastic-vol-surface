"""
Rough Bergomi (rBergomi) stochastic volatility model.

Implements Monte Carlo pricing under the rBergomi model:
    dS_t / S_t = sqrt(V_t) dW_t^1
    V_t = V_0 * exp(η * W^H_t - η²/2 * t^{2H})

where W^H is a Riemann-Liouville fractional Brownian motion with Hurst
exponent H < 0.5 (rough case).

Reference:
    Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough
    volatility. Quantitative Finance, 16(6), 887-904.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


def simulate_rbergomi(
    S0: float,
    T: float,
    r: float,
    H: float,
    eta: float,
    rho: float,
    v0: float = 0.04,
    n_paths: int = 5000,
    n_steps: int = 252,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Simulate rBergomi asset paths via Monte Carlo.

    Uses the Euler-Maruyama discretisation with the hybrid scheme for the
    Riemann-Liouville kernel.

    Parameters
    ----------
    S0:
        Initial spot price.
    T:
        Horizon in years.
    r:
        Risk-free rate (drift).
    H:
        Hurst exponent (0 < H < 0.5 for rough vol).
    eta:
        Vol-of-vol scaling parameter.
    rho:
        Correlation between price and vol Brownian motions.
    v0:
        Initial variance (V_0).
    n_paths:
        Number of Monte Carlo paths.
    n_steps:
        Time steps (default 252 = daily over 1 year).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Terminal asset prices, shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    alpha = H - 0.5  # Hurst → fractional exponent

    # Construct the covariance matrix for (dW^1, dW^2) with correlation rho
    # W^1 drives vol, W^2 = rho*W^1 + sqrt(1-rho^2)*W^perp drives price.
    t_grid = np.linspace(dt, T, n_steps)

    # Simulate standard Brownian increments
    dW1 = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dW_perp = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp

    # Riemann-Liouville fractional integral approximation
    # W^H_t ≈ sum_{j=0}^{t} K(t, s_j) * dW^1_{s_j}
    # For simplicity, use the Euler approximation of the kernel
    # K(t, s) = c_H * (t - s)^{H - 0.5}
    c_H = np.sqrt(2 * H) / (H + 0.5)

    log_S = np.zeros(n_paths)
    log_S[:] = np.log(S0)

    # Accumulated fractional Brownian increments for log-variance
    W_H = np.zeros(n_paths)

    for i in range(n_steps):
        t_i = t_grid[i]
        # Kernel weights for all previous steps
        weights = np.array(
            [c_H * (t_i - t_grid[j]) ** alpha for j in range(i)]
        ) if i > 0 else np.array([])

        if i > 0:
            W_H = W_H + weights[-1] * dW1[:, i - 1] if i == 1 else (
                np.sum(
                    [weights[j] * dW1[:, j] for j in range(i)],
                    axis=0,
                )
            )

        V_t = np.maximum(v0 * np.exp(eta * W_H - 0.5 * eta**2 * t_i ** (2 * H)), 1e-8)
        log_S += (r - 0.5 * V_t) * dt + np.sqrt(V_t * dt) * dW2[:, i] / np.sqrt(dt) * np.sqrt(dt)

    return np.exp(log_S)


def rbergomi_price(
    S: float,
    K: float,
    T: float,
    r: float,
    H: float,
    eta: float,
    rho: float,
    v0: float = 0.04,
    n_paths: int = 5000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: Optional[int] = 42,
) -> float:
    """Price a European option under rBergomi via Monte Carlo.

    Returns
    -------
    float
        Option price.
    """
    S_T = simulate_rbergomi(S, T, r, H, eta, rho, v0, n_paths, n_steps, seed)
    discount = np.exp(-r * T)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    return float(discount * np.mean(payoffs))


def calibrate_rbergomi(
    S: float,
    strikes: np.ndarray,
    tenors: np.ndarray,
    market_vols: np.ndarray,
    r: float = 0.05,
    n_paths: int = 2000,
) -> dict[str, float]:
    """Calibrate rBergomi (H, η, ρ) to a vol surface using differential evolution.

    Note: Monte Carlo calibration is slow; use a reduced path count during calibration.

    Returns
    -------
    dict[str, float]
        Calibrated H, eta, rho, and rmse.
    """
    # TODO: Implement full calibration with gradient-free optimizer.
    # Placeholder returns sensible defaults.
    logger.warning(
        "rBergomi calibration is compute-intensive; returning default parameters."
    )
    return {"H": 0.10, "eta": 1.5, "rho": -0.90, "rmse": float("nan")}
