"""Arbitrage filters for volatility surfaces.

Implements:
- Butterfly (convexity) filter based on Gatheral's density g(k) >= 0
- Calendar (monotonicity) filter: total variance must be non-decreasing in tenor
"""

from __future__ import annotations

import numpy as np


def _gatheral_density(
    k: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Compute Gatheral's risk-neutral density proxy g(k).

    For a twice-differentiable total variance surface w(k, T) at fixed T:

        g(k) = (1 - k*w'/(2w))^2 - w'^2/4*(1/w + 1/4) + w''/2

    Here w' and w'' are the first and second derivatives of total variance
    with respect to log-moneyness k.
    """
    # Numerical derivatives
    dw = np.gradient(w, k)
    d2w = np.gradient(dw, k)

    term1 = (1.0 - k * dw / (2.0 * np.maximum(w, 1e-12))) ** 2
    term2 = dw ** 2 / 4.0 * (1.0 / np.maximum(w, 1e-12) + 0.25)
    term3 = d2w / 2.0
    return term1 - term2 + term3


def butterfly_arbitrage_filter(
    strikes: np.ndarray,
    ivs: np.ndarray,
    F: float,
    T: float,
) -> tuple[bool, np.ndarray]:
    """Check for butterfly (convexity) arbitrage on a single tenor slice.

    Parameters
    ----------
    strikes : np.ndarray  Strike prices
    ivs : np.ndarray      Implied volatilities (same length as strikes)
    F : float             Forward price for this tenor
    T : float             Time-to-expiry in years

    Returns
    -------
    is_arbitrage_free : bool
        True if the slice is free of butterfly arbitrage (g >= 0 everywhere).
    g : np.ndarray
        The Gatheral density array; negative values flag arbitrage.
    """
    k = np.log(strikes / F)
    sort_idx = np.argsort(k)
    k_sorted = k[sort_idx]
    iv_sorted = ivs[sort_idx]

    # Total variance
    w = iv_sorted ** 2 * T

    if len(w) < 3:
        return True, np.zeros_like(w)

    g = _gatheral_density(k_sorted, w)
    is_free = bool(np.all(g >= -1e-8))
    return is_free, g


def calendar_arbitrage_filter(
    total_variances_by_tenor: dict[float, np.ndarray],
) -> tuple[bool, list[tuple[float, float]]]:
    """Check for calendar arbitrage across tenors.

    Calendar arbitrage occurs when total variance w(k, T1) > w(k, T2) for T1 < T2
    at any log-moneyness k.

    Parameters
    ----------
    total_variances_by_tenor : dict
        Mapping of tenor (float, in years) to 1-D array of total variances
        on a *common* log-moneyness grid.

    Returns
    -------
    is_arbitrage_free : bool
    violations : list of (T_earlier, T_later) tenor pairs that violate monotonicity
    """
    tenors = sorted(total_variances_by_tenor.keys())
    violations: list[tuple[float, float]] = []

    for i in range(len(tenors) - 1):
        T1, T2 = tenors[i], tenors[i + 1]
        w1 = total_variances_by_tenor[T1]
        w2 = total_variances_by_tenor[T2]
        if np.any(w1 > w2 + 1e-8):
            violations.append((T1, T2))

    return len(violations) == 0, violations


def remove_calendar_violations(
    total_variances_by_tenor: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    """Return a copy with calendar arbitrage removed by isotonic regression."""
    from scipy.optimize import minimize

    tenors = sorted(total_variances_by_tenor.keys())
    n_k = len(next(iter(total_variances_by_tenor.values())))

    # For each moneyness point, enforce monotonicity via projection
    cleaned: dict[float, np.ndarray] = {T: total_variances_by_tenor[T].copy() for T in tenors}

    for j in range(n_k):
        col = np.array([cleaned[T][j] for T in tenors])
        # Simple isotonic (PAVA): cumulative max
        mono = np.maximum.accumulate(col)
        for i, T in enumerate(tenors):
            cleaned[T][j] = mono[i]

    return cleaned
