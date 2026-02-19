"""Custom losses: weighted MSE (ATM 3×) and arbitrage penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mse_loss(
    pred: torch.Tensor,    # (batch, n_k, n_t)
    target: torch.Tensor,  # (batch, n_k, n_t)
    atm_weight: float = 3.0,
    atm_k_idx: int | None = None,
) -> torch.Tensor:
    """MSE loss with elevated weight on ATM region (centre of log-moneyness axis).

    Parameters
    ----------
    pred, target : (batch, n_k, n_t)
    atm_weight   : weight multiplier for ATM strikes
    atm_k_idx    : index of ATM in log-moneyness axis (default: n_k//2)
    """
    n_k = pred.shape[1]
    if atm_k_idx is None:
        atm_k_idx = n_k // 2

    weights = torch.ones_like(pred)
    weights[:, atm_k_idx, :] = atm_weight

    loss = (weights * (pred - target) ** 2).mean()
    return loss


def arbitrage_penalty(
    pred: torch.Tensor,    # (batch, n_k, n_t)
    calendar_lambda: float = 1.0,
    butterfly_lambda: float = 1.0,
) -> torch.Tensor:
    """Soft arbitrage penalty.

    Calendar: total variance must be non-decreasing in tenor (axis=-1).
    Butterfly: IV must be convex in log-moneyness (axis=-2).
    """
    # Calendar arbitrage: penalise where IV[..., t] > IV[..., t+1]
    # Use total var w = IV^2 * T ≈ IV^2 * t_idx (relative scale)
    w = pred ** 2
    # diff along tenor axis: w[:, :, t+1] - w[:, :, t]
    calendar_diff = w[:, :, 1:] - w[:, :, :-1]
    calendar_viol = F.relu(-calendar_diff)   # violations (positive where w decreases)
    cal_penalty = calendar_lambda * calendar_viol.mean()

    # Butterfly / convexity: second derivative of IV in k direction >= 0
    d2iv = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    butterfly_viol = F.relu(-d2iv)
    but_penalty = butterfly_lambda * butterfly_viol.mean()

    return cal_penalty + but_penalty
