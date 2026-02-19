"""
AIC / BIC model comparison across parametric stochastic vol models.

Selects the best-fitting model per surface snapshot using information criteria
to balance fit quality against parameter complexity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Number of free parameters per model
_MODEL_PARAMS: dict[str, int] = {
    "sabr": 3,        # α, ρ, ν (β fixed)
    "heston": 5,      # v₀, κ, θ, σ, ρ
    "rough_bergomi": 3,  # H, η, ρ
    "svi": 5,         # a, b, ρ, m, σ (per slice)
}


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion: AIC = 2k - 2*ln(L)."""
    return 2 * n_params - 2 * log_likelihood


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: BIC = k*ln(n) - 2*ln(L)."""
    return n_params * np.log(n_obs) - 2 * log_likelihood


def rmse_to_log_likelihood(rmse: float, n_obs: int) -> float:
    """Convert RMSE to approximate log-likelihood under Gaussian noise.

    Assumes residuals ~ N(0, σ²) with σ² = RMSE².
    log L = -n/2 * (log(2π) + log(σ²) + 1)
    """
    if rmse <= 0 or not np.isfinite(rmse):
        return -1e10
    sigma2 = rmse**2
    return -n_obs / 2.0 * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)


def select_best_model(
    calibration_results: dict[str, Any],
    n_obs: int,
    criterion: str = "aic",
) -> str:
    """Select the best parametric model by AIC or BIC.

    Parameters
    ----------
    calibration_results:
        Output from :func:`~src.models.calibration.calibrate_surface_snapshot`.
        Each model entry must contain an ``"rmse"`` key.
    n_obs:
        Number of observed surface points used in calibration.
    criterion:
        ``"aic"`` or ``"bic"`` (default ``"aic"``).

    Returns
    -------
    str
        Name of the best-fitting model.
    """
    scores: dict[str, float] = {}

    for model_name, result in calibration_results.items():
        rmse = result.get("rmse", float("nan"))
        if not np.isfinite(rmse) or rmse <= 0:
            continue
        k = _MODEL_PARAMS.get(model_name, 5)
        ll = rmse_to_log_likelihood(rmse, n_obs)
        if criterion == "aic":
            scores[model_name] = compute_aic(ll, k)
        else:
            scores[model_name] = compute_bic(ll, k, n_obs)

    if not scores:
        logger.warning("No valid calibration results for model selection.")
        return "sabr"  # Default fallback

    best = min(scores, key=lambda x: scores[x])
    logger.info(
        "Model selection (%s): %s (score=%.4f)", criterion.upper(), best, scores[best]
    )
    return best


def compare_models(
    calibration_results: dict[str, Any],
    n_obs: int,
) -> dict[str, dict[str, float]]:
    """Compute AIC and BIC for all calibrated models.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from model name to ``{"aic": ..., "bic": ..., "rmse": ...}``.
    """
    comparison: dict[str, dict[str, float]] = {}

    for model_name, result in calibration_results.items():
        rmse = result.get("rmse", float("nan"))
        k = _MODEL_PARAMS.get(model_name, 5)
        ll = rmse_to_log_likelihood(rmse, n_obs) if np.isfinite(rmse) else float("nan")
        comparison[model_name] = {
            "rmse": float(rmse),
            "aic": compute_aic(ll, k) if np.isfinite(ll) else float("nan"),
            "bic": compute_bic(ll, k, n_obs) if np.isfinite(ll) else float("nan"),
        }

    return comparison
