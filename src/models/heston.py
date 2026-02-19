"""
Heston stochastic volatility model.

Implements:
- Characteristic function of the log price under the Heston model.
- FFT-based European option pricing (Carr-Madan 1999).
- Calibration of (v₀, κ, θ, σ, ρ) to a vol surface.

Reference:
    Heston, S.L. (1993). A closed-form solution for options with
    stochastic volatility. Review of Financial Studies, 6(2), 327-343.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

logger = logging.getLogger(__name__)


def heston_char_fn(
    u: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> complex:
    """Heston characteristic function for log(S_T / S_0).

    Parameters
    ----------
    u:
        Complex frequency.
    S, K, T, r:
        Spot, strike, tenor, rate.
    v0:
        Initial variance.
    kappa:
        Mean-reversion speed.
    theta:
        Long-run variance.
    sigma:
        Vol-of-vol.
    rho:
        Correlation (spot, vol).
    """
    i = complex(0, 1)
    x = np.log(S / K)

    d = np.sqrt((rho * sigma * i * u - kappa) ** 2 + sigma**2 * (i * u + u**2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)

    exp_dT = np.exp(-d * T)
    C = r * i * u * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * i * u - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (1 - exp_dT) / (1 - g * exp_dT)

    return np.exp(C + D * v0 + i * u * x)


def heston_price_fft(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    option_type: str = "call",
    n_fft: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """Price a European option using Carr-Madan FFT under the Heston model.

    Parameters
    ----------
    alpha:
        Damping factor for the FFT (Carr-Madan damping).
    eta:
        FFT step size.
    n_fft:
        Number of FFT points (power of 2).

    Returns
    -------
    float
        Option price.
    """
    lambda_ = 2 * np.pi / (n_fft * eta)
    b = n_fft * lambda_ / 2
    ku = -b + lambda_ * np.arange(n_fft)

    # Integration nodes
    v = eta * np.arange(n_fft)
    i = complex(0, 1)

    # Simpson's rule weights
    w = (eta / 3) * (3 + (-1) ** np.arange(n_fft) - (np.arange(n_fft) == 0))

    # Modified characteristic function
    psi = np.exp(-r * T) * heston_char_fn(
        v - (alpha + 1) * i, S, K, T, r, v0, kappa, theta, sigma, rho
    ) / (alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v)

    fft_input = np.exp(i * b * v) * psi * w
    fft_output = np.real(np.fft.fft(fft_input))

    call_prices = np.exp(-alpha * ku) / np.pi * fft_output

    # Interpolate at log(K/S)
    log_moneyness = np.log(K / S)
    call_price = float(np.interp(log_moneyness, ku, call_prices))
    call_price = max(call_price, max(S - K * np.exp(-r * T), 0.0))

    if option_type == "put":
        # Put-call parity
        call_price = call_price - S + K * np.exp(-r * T)

    return max(call_price, 0.0)


def heston_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    option_type: str = "call",
) -> Optional[float]:
    """Compute Heston model implied vol for a single option."""
    from src.surface.implied_vol import implied_vol as bs_iv

    price = heston_price_fft(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return bs_iv(price, S, K, T, r, option_type)


def calibrate_heston(
    S: float,
    strikes: np.ndarray,
    tenors: np.ndarray,
    market_vols: np.ndarray,
    r: float = 0.05,
    initial_params: Optional[dict] = None,
) -> dict[str, float]:
    """Calibrate Heston model parameters to a volatility surface.

    Parameters
    ----------
    S:
        Spot price.
    strikes, tenors, market_vols:
        Arrays of equal length defining the observed surface points.
    r:
        Risk-free rate.
    initial_params:
        Optional initial parameter dictionary with keys v0, kappa, theta, sigma, rho.

    Returns
    -------
    dict[str, float]
        Calibrated parameters + ``rmse``.
    """
    if initial_params is None:
        initial_params = {
            "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.30, "rho": -0.70
        }

    valid = np.isfinite(market_vols)
    strikes_v = strikes[valid]
    tenors_v = tenors[valid]
    market_vols_v = market_vols[valid]

    def _objective(params: np.ndarray) -> float:
        v0, kappa, theta, sigma, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        # Feller condition: 2κθ > σ²
        if 2 * kappa * theta <= sigma**2:
            return 1e10

        errors = []
        for K, T, mv in zip(strikes_v, tenors_v, market_vols_v):
            try:
                iv = heston_implied_vol(S, K, T, r, v0, kappa, theta, sigma, rho)
                if iv is not None:
                    errors.append((iv - mv) ** 2)
            except Exception:
                pass
        return float(np.mean(errors)) if errors else 1e10

    bounds = [
        (1e-4, 2.0),    # v0
        (1e-3, 20.0),   # kappa
        (1e-4, 2.0),    # theta
        (1e-3, 5.0),    # sigma
        (-0.999, 0.999),# rho
    ]

    x0 = [
        initial_params["v0"],
        initial_params["kappa"],
        initial_params["theta"],
        initial_params["sigma"],
        initial_params["rho"],
    ]

    result = minimize(
        _objective, x0=x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500}
    )

    v0, kappa, theta, sigma, rho = result.x
    return {
        "v0": float(v0),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "rho": float(rho),
        "rmse": float(np.sqrt(result.fun)),
    }
