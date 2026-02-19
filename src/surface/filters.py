"""
Arbitrage filters for implied volatility surfaces.

Two classical no-arbitrage conditions are checked:

1. **Butterfly (convexity)**: The call price as a function of strike must be
   convex.  Equivalently, the total implied variance w(k) = σ²(k)·T must
   satisfy a positive local variance condition (Gatheral 2006).

2. **Calendar spread (monotonicity)**: The total implied variance must be
   non-decreasing in tenor.  i.e. w(k, T₁) ≤ w(k, T₂) for T₁ < T₂.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_butterfly_arbitrage(
    iv_slice: pd.Series,
    tenor_years: float,
    tol: float = 1e-6,
) -> pd.Series:
    """Identify butterfly-arbitrage violations in a single tenor slice.

    The check uses the Gatheral (2006) condition on the local variance:

        g(k) = (1 - k·w'/(2w))² - (w'/2)²·(1/4 + 1/w) + w''/2 ≥ 0

    where w(k) = σ²(k)·T and primes are k-derivatives.

    Parameters
    ----------
    iv_slice:
        pd.Series indexed by log-moneyness, values are implied vols.
    tenor_years:
        Tenor in years (T) used to compute total variance w = σ²·T.
    tol:
        Tolerance for flagging violations (default 1e-6).

    Returns
    -------
    pd.Series
        Boolean Series (same index as ``iv_slice``); ``True`` indicates a
        butterfly-arbitrage violation at that strike.
    """
    k = np.array(iv_slice.index.astype(float))
    sigma = np.array(iv_slice.values, dtype=float)
    w = sigma**2 * tenor_years

    violations = pd.Series(False, index=iv_slice.index)

    if len(k) < 3:
        return violations  # Need at least 3 points for second derivative

    # Finite-difference first and second derivatives of w w.r.t. k
    w_prime = np.gradient(w, k)
    w_double_prime = np.gradient(w_prime, k)

    for i in range(len(k)):
        wi = w[i]
        if wi <= 0 or np.isnan(wi):
            continue
        wp = w_prime[i]
        wpp = w_double_prime[i]

        g = (1.0 - k[i] * wp / (2.0 * wi)) ** 2 - (wp / 2.0) ** 2 * (
            1.0 / 4.0 + 1.0 / wi
        ) + wpp / 2.0

        if g < -tol:
            violations.iloc[i] = True

    n_viol = violations.sum()
    if n_viol > 0:
        logger.debug(
            "Butterfly violations at T=%.4f: %d / %d strikes", tenor_years, n_viol, len(k)
        )
    return violations


def check_calendar_arbitrage(
    surface_grid: pd.DataFrame,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Identify calendar-spread arbitrage violations across tenors.

    For each log-moneyness level k, the total variance w(k, T) = σ(k,T)²·T
    must be non-decreasing in T.  A violation occurs when
    w(k, Tᵢ) > w(k, Tᵢ₊₁) + tol.

    Parameters
    ----------
    surface_grid:
        DataFrame with log-moneyness index and tenor (years) columns.
        Values are implied vols.
    tol:
        Tolerance for detecting violations.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame (same shape as ``surface_grid``); ``True`` indicates
        a calendar-arbitrage violation at that (moneyness, tenor) cell.
    """
    violations = pd.DataFrame(False, index=surface_grid.index, columns=surface_grid.columns)
    tenors = np.array(surface_grid.columns.astype(float))

    for k_idx in range(len(surface_grid)):
        row_iv = surface_grid.iloc[k_idx].values.astype(float)
        w = row_iv**2 * tenors

        for t_idx in range(len(tenors) - 1):
            w_curr = w[t_idx]
            w_next = w[t_idx + 1]
            if np.isnan(w_curr) or np.isnan(w_next):
                continue
            if w_curr > w_next + tol:
                violations.iloc[k_idx, t_idx] = True

    n_viol = violations.values.sum()
    if n_viol > 0:
        logger.debug("Calendar violations: %d cells", n_viol)
    return violations


def remove_arbitrage_violations(
    surface_grid: pd.DataFrame,
    method: str = "nan",
) -> pd.DataFrame:
    """Remove or flag cells with detected arbitrage violations.

    Parameters
    ----------
    surface_grid:
        IV surface grid (log-moneyness × tenor).
    method:
        ``"nan"`` — set violating cells to NaN (default).
        ``"clip"`` — monotone clip total variance across tenors.

    Returns
    -------
    pd.DataFrame
        Cleaned surface grid.
    """
    cleaned = surface_grid.copy()
    tenors = np.array(cleaned.columns.astype(float))

    if method == "nan":
        # Mark butterfly violations per slice
        for t_idx, T in enumerate(tenors):
            slice_iv = cleaned.iloc[:, t_idx].dropna()
            if len(slice_iv) < 3:
                continue
            bt_viol = check_butterfly_arbitrage(slice_iv, T)
            # Use loc to avoid chained assignment
            viol_index = bt_viol[bt_viol].index
            cleaned.loc[viol_index, cleaned.columns[t_idx]] = np.nan

        # Mark calendar violations
        cal_viol = check_calendar_arbitrage(cleaned)
        cleaned[cal_viol] = np.nan

    elif method == "clip":
        # Monotone clip: enforce w(T₁) ≤ w(T₂) by capping earlier tenors
        for k_idx in range(len(cleaned)):
            row_iv = cleaned.iloc[k_idx].values.astype(float)
            w = row_iv**2 * tenors
            # Cumulative maximum ensures monotonicity
            w_mono = np.maximum.accumulate(np.where(np.isnan(w), -np.inf, w))
            w_mono = np.where(w_mono == -np.inf, np.nan, w_mono)
            new_iv = np.where(tenors > 0, np.sqrt(w_mono / tenors), np.nan)
            cleaned.iloc[k_idx] = new_iv

    return cleaned
