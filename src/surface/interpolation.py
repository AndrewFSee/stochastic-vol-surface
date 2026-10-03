"""Volatility surface interpolation: SVI per-slice, cubic spline across tenors, RBF for 2D.

High-level entry point
----------------------
:func:`fit_surface` – scatter DataFrame → (k_grid, t_grid, iv_grid)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_RBF = True
except ImportError:  # pragma: no cover
    _HAS_RBF = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVI per-slice
# ---------------------------------------------------------------------------

def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral raw SVI total variance:  w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_min_variance(a: float, b: float, rho: float, sigma: float) -> float:
    """Return the global minimum of a raw-SVI slice.

    Raw SVI attains its minimum at ``k - m = -rho*sigma/sqrt(1-rho^2)``, where
    ``w = a + b*sigma*sqrt(1-rho^2)``.  Requiring this to be non-negative is
    the standard condition for ``w(k) >= 0`` at every strike.
    """
    return a + b * sigma * math.sqrt(max(1.0 - rho ** 2, 0.0))


def interpolate_svi_slice(
    log_moneyness: np.ndarray,
    total_var: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit SVI to a single tenor slice and evaluate on a dense grid.

    The fit is constrained so total variance stays non-negative everywhere
    (``a + b*sigma*sqrt(1-rho^2) >= 0``).  Without it the optimiser happily
    returns a slice that fits the quoted strikes well but goes negative in an
    unquoted wing, which then gets floored to ~0 and produces a near-zero
    implied vol on the standardised grid.

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

    # Repair only if the unconstrained fit is actually negative somewhere.
    # The constraint is deliberately NOT folded into `loss` above: a hard
    # penalty wall inside an L-BFGS-B objective breaks its finite-difference
    # gradients and degrades slices that were fitting perfectly well.
    if svi_min_variance(a, b, rho, sigma) < 0:
        logger.debug("SVI slice negative in the wing (min w=%.3e) — repairing",
                     svi_min_variance(a, b, rho, sigma))

        def sse(p):
            return float(np.sum((svi_raw(log_moneyness, *p) - total_var) ** 2))

        cons = [{
            "type": "ineq",
            "fun": lambda p: svi_min_variance(p[0], p[1], p[2], p[4]),
        }]
        # Seed from the unconstrained solution, lifted to be feasible, so the
        # constrained search starts near the shape the data actually implies.
        seed = [a - svi_min_variance(a, b, rho, sigma), b, rho, m, sigma]
        seed[0] = float(np.clip(seed[0], bounds[0][0], bounds[0][1]))

        best = None
        for start in (seed, x0):
            try:
                r2 = minimize(sse, start, method="SLSQP", bounds=bounds,
                              constraints=cons,
                              options={"maxiter": 500, "ftol": 1e-12})
            except Exception:
                continue
            if svi_min_variance(r2.x[0], r2.x[1], r2.x[2], r2.x[4]) >= 0:
                if best is None or r2.fun < best.fun:
                    best = r2

        if best is not None:
            a, b, rho, m, sigma = best.x
        else:
            # Last resort: lift the level so the slice is non-negative,
            # preserving the fitted shape.
            a = a - svi_min_variance(a, b, rho, sigma)

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


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def fit_surface(
    scatter_df,
    k_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    method: str = "svi",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an IV surface from a scatter DataFrame.

    This is a thin convenience wrapper that delegates to
    :func:`grid_builder.interpolate_to_grid` so that either module can be
    the caller's entry point.

    Parameters
    ----------
    scatter_df : pd.DataFrame
        Output of :func:`grid_builder.build_surface_grid`.
    k_grid, t_grid : array-like or None
        Defaults to ``config/surface_grid.yaml``.
    method : ``"svi"`` | ``"rbf"``

    Returns
    -------
    k_nodes, t_nodes, iv_grid
    """
    from src.surface.grid_builder import interpolate_to_grid

    return interpolate_to_grid(scatter_df, k_grid=k_grid, tenor_grid=t_grid, method=method)
