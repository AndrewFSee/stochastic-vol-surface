"""Volatility surface interpolation: SVI per-slice, cubic spline across tenors, RBF for 2D."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_RBF = True
except ImportError:  # pragma: no cover
    _HAS_RBF = False


# ---------------------------------------------------------------------------
# SVI per-slice
# ---------------------------------------------------------------------------

def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral raw SVI total variance:  w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def interpolate_svi_slice(
    log_moneyness: np.ndarray,
    total_var: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit SVI to a single tenor slice and evaluate on a dense grid.

    Returns the dense-grid total-variance array and the fitted parameter dict.
    """
    from scipy.optimize import minimize

    def loss(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e9
        w = svi_raw(log_moneyness, a, b, rho, m, sigma)
        return float(np.sum((w - total_var) ** 2))

    k_range = log_moneyness.max() - log_moneyness.min()
    x0 = [total_var.mean(), 0.1, -0.5, log_moneyness.mean(), k_range / 4]
    bounds = [(-1, 1), (0, 2), (-0.999, 0.999), (-2, 2), (1e-4, 2)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-12})

    a, b, rho, m, sigma = res.x
    w_fit = svi_raw(log_moneyness, a, b, rho, m, sigma)
    params = dict(a=a, b=b, rho=rho, m=m, sigma=sigma)
    return w_fit, params


# ---------------------------------------------------------------------------
# Cubic spline across tenors
# ---------------------------------------------------------------------------

def interpolate_across_tenors(
    tenors: np.ndarray,
    slices: np.ndarray,  # shape (n_tenors, n_k)
    new_tenors: np.ndarray,
) -> np.ndarray:
    """Cubic spline interpolation of IV slices across tenor dimension.

    Parameters
    ----------
    tenors : (n_tenors,)
    slices : (n_tenors, n_k)  – IV values per tenor per log-moneyness
    new_tenors : (m,)

    Returns
    -------
    (m, n_k) interpolated IV array
    """
    n_k = slices.shape[1]
    out = np.empty((len(new_tenors), n_k))
    for j in range(n_k):
        cs = CubicSpline(tenors, slices[:, j], extrapolate=False)
        out[:, j] = cs(new_tenors)
    return out


# ---------------------------------------------------------------------------
# RBF 2-D interpolation
# ---------------------------------------------------------------------------

def rbf_interpolate_surface(
    points: np.ndarray,   # (N, 2)  columns: [log_moneyness, tenor]
    values: np.ndarray,   # (N,)    implied vols
    k_grid: np.ndarray,   # (n_k,)
    t_grid: np.ndarray,   # (n_t,)
    smoothing: float = 1e-3,
) -> np.ndarray:
    """RBF (thin-plate spline) interpolation onto a regular grid.

    Returns iv_grid of shape (n_k, n_t).
    """
    if not _HAS_RBF:
        raise ImportError("scipy >= 1.7 required for RBFInterpolator")

    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=smoothing)
    KK, TT = np.meshgrid(k_grid, t_grid, indexing="ij")
    query = np.column_stack([KK.ravel(), TT.ravel()])
    iv_flat = rbf(query)
    return iv_flat.reshape(len(k_grid), len(t_grid))


# ---------------------------------------------------------------------------
# Convenience: build a RegularGridInterpolator from a grid
# ---------------------------------------------------------------------------

def build_regular_grid_interpolator(
    k_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_grid: np.ndarray,
    bounds_error: bool = False,
    fill_value: float | None = None,
) -> RegularGridInterpolator:
    """Wrap a (k, T) IV grid in a RegularGridInterpolator."""
    return RegularGridInterpolator(
        (k_grid, t_grid),
        iv_grid,
        method="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
