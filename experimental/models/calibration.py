"""Unified calibration engine: scipy L-BFGS-B + differential evolution."""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


def calibrate(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    x0: np.ndarray | None = None,
    use_global: bool = True,
    de_maxiter: int = 300,
    local_maxiter: int = 1000,
    tol: float = 1e-10,
    seed: int = 42,
) -> dict[str, object]:
    """Generic calibration engine.

    1. (Optional) global search with differential evolution
    2. Local refinement with L-BFGS-B

    Parameters
    ----------
    objective : callable  f(params) → scalar loss
    bounds    : list of (lo, hi) per parameter
    x0        : initial guess (used if use_global=False)
    use_global: run differential evolution first

    Returns
    -------
    dict with keys: x (np.ndarray), fun (float), success (bool), message (str)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if use_global:
            de_res = differential_evolution(
                objective,
                bounds,
                maxiter=de_maxiter,
                tol=tol,
                seed=seed,
                workers=1,
                polish=False,
            )
            x_init = de_res.x
        else:
            x_init = x0 if x0 is not None else np.array([(lo + hi) / 2 for lo, hi in bounds])

        local_res = minimize(
            objective,
            x_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": local_maxiter, "ftol": tol},
        )

    return dict(
        x=local_res.x,
        fun=float(local_res.fun),
        success=bool(local_res.success),
        message=local_res.message,
    )


def rmse(model_vals: np.ndarray, market_vals: np.ndarray) -> float:
    """Root-mean-square error between model and market values."""
    return float(np.sqrt(np.mean((model_vals - market_vals) ** 2)))


def weighted_rmse(
    model_vals: np.ndarray,
    market_vals: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted RMSE; if weights=None uses uniform weights."""
    if weights is None:
        return rmse(model_vals, market_vals)
    w = weights / weights.sum()
    return float(np.sqrt(np.sum(w * (model_vals - market_vals) ** 2)))
