"""
Unified calibration engine for all parametric stochastic volatility models.

Provides a single entry point :func:`calibrate_surface_snapshot` that fits
all configured models to a surface snapshot and returns parameters + errors.

Uses scipy L-BFGS-B as the primary optimizer with differential evolution as
a fallback for non-convex objective landscapes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Supported model names
SUPPORTED_MODELS = ("sabr", "heston", "svi")


def calibrate_surface_snapshot(
    surface_grid: pd.DataFrame,
    spot: float,
    risk_free_rate: float = 0.05,
    models: tuple[str, ...] = SUPPORTED_MODELS,
) -> dict[str, Any]:
    """Calibrate multiple models to a surface grid snapshot.

    Parameters
    ----------
    surface_grid:
        DataFrame with log-moneyness index and tenor (years) columns.
        Values are implied vols.
    spot:
        Spot price.
    risk_free_rate:
        Continuously-compounded risk-free rate (flat curve approximation).
    models:
        Tuple of model names to calibrate.

    Returns
    -------
    dict[str, Any]
        Nested dict: ``{model_name: {params..., rmse, model_vols: ndarray}}``.
    """
    results: dict[str, Any] = {}

    moneyness = np.array(surface_grid.index.astype(float))
    tenors = np.array(surface_grid.columns.astype(float))

    # Flatten to arrays for calibration
    k_flat, T_flat, iv_flat = [], [], []
    for t_idx, T in enumerate(tenors):
        for k_idx, k in enumerate(moneyness):
            iv = surface_grid.iloc[k_idx, t_idx]
            if np.isfinite(iv):
                k_flat.append(k)
                T_flat.append(T)
                iv_flat.append(iv)

    k_arr = np.array(k_flat)
    T_arr = np.array(T_flat)
    iv_arr = np.array(iv_flat)

    # Convert log-moneyness + spot → strikes
    # K = S * exp(k)  (approximate: uses spot instead of forward)
    K_arr = spot * np.exp(k_arr)

    if "sabr" in models:
        results["sabr"] = _calibrate_sabr_surface(
            spot, K_arr, T_arr, iv_arr, risk_free_rate
        )

    if "svi" in models:
        results["svi"] = _calibrate_svi_surface(k_arr, T_arr, iv_arr)

    if "heston" in models:
        results["heston"] = _calibrate_heston_surface(
            spot, K_arr, T_arr, iv_arr, risk_free_rate
        )

    return results


def _calibrate_sabr_surface(
    S: float,
    strikes: np.ndarray,
    tenors: np.ndarray,
    market_vols: np.ndarray,
    r: float,
) -> dict[str, Any]:
    """Calibrate SABR per tenor slice and aggregate."""
    from src.models.sabr import calibrate_sabr

    per_slice_params: list[dict] = []
    unique_tenors = np.unique(tenors)

    for T in unique_tenors:
        mask = tenors == T
        F = S * np.exp(r * T)  # Approximate forward
        params = calibrate_sabr(
            F=F,
            strikes=strikes[mask],
            market_vols=market_vols[mask],
            T=T,
            beta=0.5,
        )
        params["T"] = T
        per_slice_params.append(params)

    avg_rmse = np.mean([p["rmse"] for p in per_slice_params if np.isfinite(p["rmse"])])
    return {"per_slice": per_slice_params, "rmse": float(avg_rmse)}


def _calibrate_svi_surface(
    k: np.ndarray,
    tenors: np.ndarray,
    market_vols: np.ndarray,
) -> dict[str, Any]:
    """Calibrate SVI per tenor slice."""
    from src.models.svi import fit_svi_slice

    per_slice_params: list[dict] = []
    unique_tenors = np.unique(tenors)

    for T in unique_tenors:
        mask = tenors == T
        params = fit_svi_slice(k[mask], market_vols[mask], T=T)
        if params is not None:
            a, b, rho, m, sigma = params
            per_slice_params.append({"T": T, "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma})

    return {"per_slice": per_slice_params, "rmse": float("nan")}


def _calibrate_heston_surface(
    S: float,
    strikes: np.ndarray,
    tenors: np.ndarray,
    market_vols: np.ndarray,
    r: float,
) -> dict[str, Any]:
    """Calibrate Heston across the full surface."""
    from src.models.heston import calibrate_heston

    try:
        params = calibrate_heston(S, strikes, tenors, market_vols, r)
    except Exception as exc:
        logger.warning("Heston calibration failed: %s", exc)
        params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.30, "rho": -0.70,
                  "rmse": float("nan")}
    return params
