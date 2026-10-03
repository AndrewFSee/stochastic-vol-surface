"""Heston model: characteristic function, FFT pricing, and calibration."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.optimize import minimize


OptionType = Literal["call", "put"]


def heston_char_func(
    u: np.ndarray,
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    """Heston characteristic function for log-return log(S_T/K).

    Uses the form from Albrecher et al. (2007) that avoids discontinuities.
    """
    i = complex(0, 1)
    lnS = np.log(S)
    lnK = np.log(K)

    d = np.sqrt((rho * sigma * i * u - kappa) ** 2 + sigma ** 2 * (i * u + u ** 2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)

    exp_dT = np.exp(-d * T)
    C = r * i * u * T + (kappa * theta / sigma ** 2) * (
        (kappa - rho * sigma * i * u - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = (kappa - rho * sigma * i * u - d) / sigma ** 2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(C + D * v0 + i * u * lnS)


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
    option_type: OptionType = "call",
    n_fft: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """Price a European option via Carr-Madan FFT.

    Parameters
    ----------
    v0    : Initial variance
    kappa : Mean-reversion speed
    theta : Long-run variance
    sigma : Vol-of-vol
    rho   : Correlation between spot and variance
    """
    N = n_fft
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2

    j = np.arange(1, N + 1, dtype=complex)
    v_j = eta * (j - 1)

    i = complex(0, 1)
    psi = (
        np.exp(-r * T)
        / ((alpha + i * v_j) * (alpha + 1 + i * v_j))
        * heston_char_func(v_j - (alpha + 1) * i, S, K, T, r, v0, kappa, theta, sigma, rho)
    )

    # Simpson's rule weights
    w = np.ones(N, dtype=complex)
    w[0] = 1 / 3
    w[-1] = 1 / 3
    w[1:-1:2] = 4 / 3
    w[2:-2:2] = 2 / 3

    x = psi * np.exp(i * b * v_j) * w * eta
    fft_val = np.fft.fft(x)

    k_u = -b + lam * (np.arange(1, N + 1) - 1)
    log_K = np.log(K)

    idx = int(np.argmin(np.abs(k_u - log_K)))
    call_price = float(
        np.real(np.exp(-alpha * k_u[idx]) / np.pi * fft_val[idx])
    )

    if option_type == "call":
        return max(call_price, 0.0)
    else:
        # Put via put-call parity
        return max(call_price - S + K * np.exp(-r * T), 0.0)


def calibrate_heston(
    S: float,
    strikes: np.ndarray,
    market_prices: np.ndarray,
    T: float,
    r: float,
    option_types: list[str] | None = None,
    x0: np.ndarray | None = None,
) -> dict[str, float]:
    """Calibrate Heston to market prices using differential evolution + L-BFGS-B."""
    if option_types is None:
        option_types = ["call"] * len(strikes)

    if x0 is None:
        x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.7])  # v0, kappa, theta, sigma, rho

    def objective(params):
        v0, kappa, theta, sigma, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e9
        # Feller condition: 2*kappa*theta >= sigma^2
        if 2 * kappa * theta < sigma ** 2:
            return 1e9
        err = 0.0
        for K, mp, ot in zip(strikes, market_prices, option_types):
            try:
                model_p = heston_price_fft(S, K, T, r, v0, kappa, theta, sigma, rho,
                                           option_type=ot)
                err += (model_p - mp) ** 2
            except Exception:
                err += 1e6
        return err

    bounds = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (1e-4, 2.0), (-0.999, 0.999)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})

    v0, kappa, theta, sigma, rho = res.x
    return dict(v0=float(v0), kappa=float(kappa), theta=float(theta),
                sigma=float(sigma), rho=float(rho), fun=float(res.fun))
