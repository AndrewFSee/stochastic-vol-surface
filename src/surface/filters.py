"""Arbitrage filters for volatility surfaces.

Implements:
- Butterfly (convexity) filter based on Gatheral's density g(k) >= 0
- Calendar (monotonicity) filter: total variance must be non-decreasing in tenor
- DataFrame-level wrappers for the pipeline
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gatheral_density(
    k: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Compute Gatheral's risk-neutral density proxy g(k).

    For a twice-differentiable total variance surface w(k, T) at fixed T:

        g(k) = (1 - k*w'/(2w))^2 - w'^2/4*(1/w + 1/4) + w''/2

    Here w' and w'' are the first and second derivatives of total variance
    with respect to log-moneyness k.

    Callers should ensure *k* is sorted and has no duplicate values.
    """
    if len(k) < 3:
        return np.zeros(len(k))
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


# ---------------------------------------------------------------------------
# DataFrame-level wrappers for the surface construction pipeline
# ---------------------------------------------------------------------------

def remove_butterfly_violations(
    df: pd.DataFrame,
    T_col: str = "T",
    k_col: str = "log_moneyness",
    iv_col: str = "implied_volatility",
    forward_col: str = "forward",
    strike_col: str = "strike",
    tol: float = -1e-6,
) -> pd.DataFrame:
    """Remove strikes that violate butterfly arbitrage within each tenor slice.

    For each unique tenor in *df*, the Gatheral density g(k) is computed.
    Strikes where g(k) < *tol* are removed.

    Returns
    -------
    pd.DataFrame  – filtered copy of *df* (same columns, fewer rows).
    """
    keep_mask = np.ones(len(df), dtype=bool)

    for T_val, grp in df.groupby(T_col):
        if len(grp) < 3:
            continue  # too few points to check

        idx = grp.index
        k = grp[k_col].to_numpy()
        iv = grp[iv_col].to_numpy()
        T = float(T_val)

        sort_order = np.argsort(k)
        k_sorted = k[sort_order]
        iv_sorted = iv[sort_order]

        # De-duplicate strikes (keep first occurrence)
        unique_mask = np.concatenate(([True], np.diff(k_sorted) > 1e-12))
        k_uniq = k_sorted[unique_mask]
        iv_uniq = iv_sorted[unique_mask]

        if len(k_uniq) < 3:
            continue

        w_uniq = iv_uniq ** 2 * T
        g_uniq = _gatheral_density(k_uniq, w_uniq)

        # Map density back to the full (possibly duplicate) sorted array:
        # duplicates inherit the density of the unique point they collapsed to
        cum_unique = np.cumsum(unique_mask) - 1  # index into g_uniq
        g_sorted = g_uniq[cum_unique]

        bad_sorted = g_sorted < tol
        bad = np.empty_like(bad_sorted)
        bad[sort_order] = bad_sorted

        if bad.any():
            keep_mask[idx[bad]] = False
            logger.debug(
                "Butterfly filter removed %d/%d strikes at T=%.4f",
                bad.sum(), len(grp), T,
            )

    n_removed = (~keep_mask).sum()
    if n_removed:
        logger.info("Butterfly arbitrage filter removed %d rows total.", n_removed)
    return df.loc[keep_mask].reset_index(drop=True)


def apply_calendar_filter_grid(
    k_grid: np.ndarray,
    tenor_grid: np.ndarray,
    iv_grid: np.ndarray,
) -> np.ndarray:
    """Enforce calendar monotonicity on an already-interpolated grid.

    Operates on total variance w = σ²·T and enforces w non-decreasing in T
    for each moneyness column via isotonic (PAVA) projection.

    Parameters
    ----------
    k_grid : (n_k,)
    tenor_grid : (n_t,)
    iv_grid : (n_k, n_t) – implied volatilities

    Returns
    -------
    iv_clean : (n_k, n_t)
    """
    n_k, n_t = iv_grid.shape
    tv = iv_grid ** 2 * tenor_grid[np.newaxis, :]  # total variance

    tv_dict: dict[float, np.ndarray] = {}
    for j, T in enumerate(tenor_grid):
        tv_dict[T] = tv[:, j]

    tv_clean = remove_calendar_violations(tv_dict)

    iv_clean = np.empty_like(iv_grid)
    for j, T in enumerate(tenor_grid):
        w = np.maximum(tv_clean[T], 1e-12)
        iv_clean[:, j] = np.sqrt(w / max(T, 1e-6))
    return iv_clean


def apply_arbitrage_filters(
    df: pd.DataFrame,
    *,
    remove_butterfly: bool = True,
    tol: float = -1e-6,
) -> pd.DataFrame:
    """Single entry-point: remove butterfly violations from a scatter DataFrame.

    Calendar arbitrage is handled *after* grid interpolation via
    :func:`apply_calendar_filter_grid`.
    """
    if remove_butterfly:
        df = remove_butterfly_violations(df, tol=tol)
    return df
