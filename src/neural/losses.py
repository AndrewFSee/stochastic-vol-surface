"""
Custom loss functions for neural SDE vol surface learning.

Two main components:
1. **Weighted MSE** — ATM options receive 3× weight (most liquid, most important).
2. **Arbitrage penalty** — penalises butterfly and calendar violations in the
   predicted surface.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    moneyness_grid: torch.Tensor,
    atm_weight: float = 3.0,
    atm_threshold: float = 0.05,
) -> torch.Tensor:
    """Weighted MSE with higher weight on near-ATM options.

    Parameters
    ----------
    pred:
        Predicted IV surface, shape (batch, n_moneyness, n_tenors).
    target:
        Target IV surface, same shape.
    moneyness_grid:
        Log-moneyness values, shape (n_moneyness,).
    atm_weight:
        Multiplier for ATM region (|k| ≤ atm_threshold).
    atm_threshold:
        Half-width of the ATM band in log-moneyness units.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    weights = torch.ones_like(moneyness_grid)
    atm_mask = moneyness_grid.abs() <= atm_threshold
    weights[atm_mask] = atm_weight

    # Broadcast weights over (batch, n_moneyness, n_tenors)
    weights = weights.unsqueeze(0).unsqueeze(-1)

    # Mask NaN targets
    valid = ~torch.isnan(target)
    pred_masked = pred[valid]
    target_masked = target[valid]
    weight_masked = weights.expand_as(target)[valid]

    if pred_masked.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    sq_err = weight_masked * (pred_masked - target_masked) ** 2
    return sq_err.mean()


def butterfly_arbitrage_penalty(
    pred: torch.Tensor,
    tenor_grid: torch.Tensor,
    moneyness_grid: torch.Tensor,
    lambda_penalty: float = 0.1,
) -> torch.Tensor:
    """Penalise butterfly-arbitrage violations in the predicted surface.

    Approximates the Gatheral local variance condition:
        g(k) = (1 - k*w'/(2w))² - (w'/2)²*(1/4 + 1/w) + w''/2 ≥ 0

    Uses finite differences for w' and w''.

    Parameters
    ----------
    pred:
        Predicted IV surface, shape (batch, n_moneyness, n_tenors).
    tenor_grid:
        Tenor values in years, shape (n_tenors,).
    moneyness_grid:
        Log-moneyness values, shape (n_moneyness,).
    lambda_penalty:
        Penalty scaling factor.

    Returns
    -------
    torch.Tensor
        Scalar penalty ≥ 0.
    """
    # Total variance w = σ² * T
    T = tenor_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_tenors)
    w = pred**2 * T  # (batch, n_moneyness, n_tenors)

    # Approximate dk = mean spacing of moneyness grid
    dk = (moneyness_grid[-1] - moneyness_grid[0]) / (len(moneyness_grid) - 1)

    # First and second derivatives via central differences
    w_prime = torch.gradient(w, dim=1)[0] / dk
    w_double_prime = torch.gradient(w_prime, dim=1)[0] / dk

    eps = 1e-8
    k = moneyness_grid.unsqueeze(0).unsqueeze(-1)  # (1, n_moneyness, 1)

    g = (1.0 - k * w_prime / (2.0 * (w + eps))) ** 2 \
        - (w_prime / 2.0) ** 2 * (1.0 / 4.0 + 1.0 / (w + eps)) \
        + w_double_prime / 2.0

    # Penalise negative g values
    penalty = torch.relu(-g).mean()
    return lambda_penalty * penalty


def calendar_arbitrage_penalty(
    pred: torch.Tensor,
    tenor_grid: torch.Tensor,
    lambda_penalty: float = 0.1,
) -> torch.Tensor:
    """Penalise calendar-spread violations: w(k,T1) > w(k,T2) for T1 < T2.

    Parameters
    ----------
    pred:
        Predicted IV surface, shape (batch, n_moneyness, n_tenors).
    tenor_grid:
        Tenor values in years, shape (n_tenors,).
    lambda_penalty:
        Penalty scaling factor.

    Returns
    -------
    torch.Tensor
        Scalar penalty ≥ 0.
    """
    T = tenor_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_tenors)
    w = pred**2 * T

    # Difference between consecutive tenors (should be ≥ 0)
    dw = w[:, :, 1:] - w[:, :, :-1]
    penalty = torch.relu(-dw).mean()
    return lambda_penalty * penalty


class VolSurfaceLoss(nn.Module):
    """Combined loss: weighted MSE + arbitrage penalties.

    Parameters
    ----------
    moneyness_grid:
        Tensor of log-moneyness values.
    tenor_grid:
        Tensor of tenor values in years.
    atm_weight:
        ATM weighting for MSE.
    lambda_butterfly:
        Butterfly penalty coefficient.
    lambda_calendar:
        Calendar penalty coefficient.
    """

    def __init__(
        self,
        moneyness_grid: torch.Tensor,
        tenor_grid: torch.Tensor,
        atm_weight: float = 3.0,
        lambda_butterfly: float = 0.1,
        lambda_calendar: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("moneyness_grid", moneyness_grid)
        self.register_buffer("tenor_grid", tenor_grid)
        self.atm_weight = atm_weight
        self.lambda_butterfly = lambda_butterfly
        self.lambda_calendar = lambda_calendar

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = weighted_mse_loss(pred, target, self.moneyness_grid, self.atm_weight)  # type: ignore[arg-type]
        bt_pen = butterfly_arbitrage_penalty(pred, self.tenor_grid, self.moneyness_grid, self.lambda_butterfly)  # type: ignore[arg-type]
        cal_pen = calendar_arbitrage_penalty(pred, self.tenor_grid, self.lambda_calendar)  # type: ignore[arg-type]
        return mse + bt_pen + cal_pen
