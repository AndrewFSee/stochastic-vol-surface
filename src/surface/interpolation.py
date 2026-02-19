"""
SVI per-slice, cubic spline across tenors, and RBF 2D interpolation.

Provides utilities to fill NaN cells in the standardised surface grid
via various interpolation strategies.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, CubicSpline

logger = logging.getLogger(__name__)


def interpolate_slice(
    slice_iv: pd.Series,
    target_moneyness: np.ndarray,
    method: Literal["linear", "cubic", "svi"] = "cubic",
) -> np.ndarray:
    """Interpolate a single tenor slice to the target moneyness grid.

    Parameters
    ----------
    slice_iv:
        pd.Series of observed implied vols indexed by log-moneyness.
    target_moneyness:
        Target log-moneyness array.
    method:
        ``"linear"``, ``"cubic"``, or ``"svi"`` (fits an SVI smile).

    Returns
    -------
    np.ndarray
        Interpolated implied vols at ``target_moneyness`` (NaN where
        outside range and extrapolation is disabled).
    """
    clean = slice_iv.dropna()
    if len(clean) < 2:
        return np.full(len(target_moneyness), np.nan)

    k_obs = clean.index.astype(float).values
    iv_obs = clean.values.astype(float)
    sort_idx = np.argsort(k_obs)
    k_obs = k_obs[sort_idx]
    iv_obs = iv_obs[sort_idx]

    if method == "svi":
        try:
            from src.models.svi import fit_svi_slice

            params = fit_svi_slice(k_obs, iv_obs)
            if params is not None:
                from src.models.svi import svi_smile

                return svi_smile(target_moneyness, *params)
        except Exception as exc:
            logger.debug("SVI fit failed; falling back to cubic: %s", exc)

    # Cubic spline (with flat extrapolation outside range)
    if len(k_obs) >= 4 and method == "cubic":
        cs = CubicSpline(k_obs, iv_obs, bc_type="natural", extrapolate=False)
        result = cs(target_moneyness)
        # Flat extrapolation for out-of-range points
        result = np.where(target_moneyness < k_obs[0], iv_obs[0], result)
        result = np.where(target_moneyness > k_obs[-1], iv_obs[-1], result)
        return np.clip(result, 1e-4, None)

    # Linear fallback
    return np.interp(target_moneyness, k_obs, iv_obs)


def interpolate_across_tenors(
    surface_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Fill NaN cells by cubic spline interpolation across the tenor axis.

    For each log-moneyness row, fit a cubic spline to the observed (non-NaN)
    tenor values and evaluate at the grid points.

    Parameters
    ----------
    surface_grid:
        DataFrame with log-moneyness index and tenor columns.

    Returns
    -------
    pd.DataFrame
        Filled surface grid.
    """
    filled = surface_grid.copy()
    tenors = np.array(surface_grid.columns.astype(float))

    for k_idx in range(len(filled)):
        row = filled.iloc[k_idx].values.astype(float)
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue

        t_obs = tenors[valid]
        iv_obs = row[valid]

        if valid.sum() >= 4:
            cs = CubicSpline(t_obs, iv_obs, bc_type="natural", extrapolate=False)
            interp = cs(tenors)
            # Flat extrapolation
            interp = np.where(tenors < t_obs[0], iv_obs[0], interp)
            interp = np.where(tenors > t_obs[-1], iv_obs[-1], interp)
        else:
            interp = np.interp(tenors, t_obs, iv_obs)

        interp = np.clip(interp, 1e-4, None)
        # Only fill NaN cells; keep existing data intact
        filled.iloc[k_idx] = np.where(np.isnan(row), interp, row)

    return filled


def interpolate_rbf_2d(
    surface_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Fill NaN cells using a 2-D RBF interpolation.

    Uses all observed (moneyness, tenor) → IV data points to fit a
    thin-plate-spline RBF and evaluates it on the full grid.

    Parameters
    ----------
    surface_grid:
        DataFrame with log-moneyness index and tenor columns.

    Returns
    -------
    pd.DataFrame
        Filled surface grid.
    """
    filled = surface_grid.copy()
    moneyness = np.array(surface_grid.index.astype(float))
    tenors = np.array(surface_grid.columns.astype(float))

    # Collect observed points
    k_list, t_list, iv_list = [], [], []
    for k_idx, k in enumerate(moneyness):
        for t_idx, t in enumerate(tenors):
            iv = surface_grid.iloc[k_idx, t_idx]
            if not np.isnan(iv):
                k_list.append(k)
                t_list.append(t)
                iv_list.append(iv)

    if len(iv_list) < 4:
        return filled

    points = np.column_stack([k_list, t_list])
    values = np.array(iv_list)

    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=0.001)

    # Evaluate on full grid
    kk, tt = np.meshgrid(moneyness, tenors, indexing="ij")
    query = np.column_stack([kk.ravel(), tt.ravel()])
    interp = rbf(query).reshape(len(moneyness), len(tenors))
    interp = np.clip(interp, 1e-4, None)

    original = surface_grid.values.astype(float)
    result = np.where(np.isnan(original), interp, original)
    return pd.DataFrame(result, index=surface_grid.index, columns=surface_grid.columns)
