"""rough-Bergomi (rBergomi) model: Monte Carlo pricing and calibration stub."""

from __future__ import annotations

import warnings

import numpy as np


def simulate_rough_bergomi(
    S0: float,
    xi0: float,
    eta: float,
    rho: float,
    H: float,
    T: float,
    r: float = 0.0,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rBergomi paths using the Euler-Maruyama scheme.

    Returns
    -------
    S : np.ndarray  shape (n_paths, n_steps+1)  – simulated spot paths
    V : np.ndarray  shape (n_paths, n_steps+1)  – simulated variance paths
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Volterra kernel exponent for rough vol
    alpha = H - 0.5  # H = 0.1 → α = -0.4

    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    V[:, 0] = xi0

    # Correlated Brownian increments
    dW1 = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))
    dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

    # Rough volatility: V(t) ≈ xi0 * exp(eta * W^H(t) - 0.5*eta^2*t^{2H})
    # Riemann-Liouville kernel approximation (simplified)
    for j in range(n_steps):
        t_j = t[j + 1]
        # Approximation of fractional Brownian motion increment
        fbm_incr = np.sum(
            [(t_j - t[k]) ** alpha * dW2[:, k] for k in range(j + 1)], axis=0
        ) * dt ** 0  # kernel already embedded in sum

        V[:, j + 1] = xi0 * np.exp(
            eta * fbm_incr - 0.5 * eta ** 2 * t_j ** (2 * H)
        )
        V[:, j + 1] = np.maximum(V[:, j + 1], 1e-8)

        S[:, j + 1] = S[:, j] * np.exp(
            (r - 0.5 * V[:, j]) * dt + np.sqrt(V[:, j]) * dW1[:, j]
        )

    return S, V


def price_option_rough_bergomi(
    S0: float,
    K: float,
    T: float,
    r: float,
    xi0: float,
    eta: float,
    rho: float,
    H: float = 0.1,
    n_paths: int = 10_000,
    n_steps: int = 100,
    option_type: str = "call",
    seed: int | None = 42,
) -> float:
    """Price a European option under rBergomi via Monte Carlo."""
    S, _ = simulate_rough_bergomi(S0, xi0, eta, rho, H, T, r, n_paths, n_steps, seed)
    S_T = S[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    price = np.exp(-r * T) * np.mean(payoff)
    return float(price)


def calibrate_rough_bergomi(
    S0: float,
    strikes: np.ndarray,
    market_prices: np.ndarray,
    T: float,
    r: float = 0.0,
    H: float = 0.1,
    x0: np.ndarray | None = None,
    n_paths: int = 5_000,
    n_steps: int = 50,
) -> dict[str, float]:
    """Stub calibration using Nelder-Mead (slow – reduce n_paths for speed)."""
    from scipy.optimize import minimize

    if x0 is None:
        x0 = np.array([0.3, 0.3, -0.7])  # xi0, eta, rho

    def objective(params):
        xi0, eta, rho = params
        if xi0 <= 0 or eta <= 0 or abs(rho) >= 1:
            return 1e9
        err = 0.0
        for K, mp in zip(strikes, market_prices):
            try:
                model_p = price_option_rough_bergomi(
                    S0, K, T, r, xi0, eta, rho, H, n_paths, n_steps, seed=42
                )
                err += (model_p - mp) ** 2
            except Exception:
                err += 1e6
        return err

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-4})

    xi0, eta, rho = res.x
    return dict(xi0=float(xi0), eta=float(eta), rho=float(rho), H=float(H), fun=float(res.fun))
