"""Full Neural Vol Model: parametric IV + Neural SDE residuals."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from src.neural.neural_sde import NeuralSDE
from src.neural.encoder import SurfaceEncoder
from src.neural.decoder import SurfaceDecoder


class NeuralVolModel(nn.Module):
    """Learns IV surface residuals: market_iv - parametric_iv.

    Architecture
    ------------
    1. SurfaceEncoder   : (batch, 1, n_k, n_t) → (batch, latent_dim)
    2. NeuralSDE        : (batch, latent_dim) × time_steps → (batch, T, latent_dim)
    3. SurfaceDecoder   : (batch, latent_dim) → (batch, n_k * n_t)
    """

    def __init__(
        self,
        n_k: int,
        n_t: int,
        latent_dim: int = 16,
        state_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_k = n_k
        self.n_t = n_t
        self.latent_dim = latent_dim

        self.encoder = SurfaceEncoder(n_k=n_k, n_t=n_t, latent_dim=latent_dim)
        self.sde = NeuralSDE(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = SurfaceDecoder(latent_dim=state_dim, n_k=n_k, n_t=n_t, hidden_dim=hidden_dim)

        # Project encoder output to SDE state dim
        self.enc_proj = nn.Linear(latent_dim, state_dim)

    def forward(
        self,
        surface_snapshots: torch.Tensor,  # (batch, seq, 1, n_k, n_t)
        ts: torch.Tensor,                 # (seq,) forecast time grid
        parametric_iv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict residual IV surfaces.

        Parameters
        ----------
        surface_snapshots : tensor (batch, seq, 1, n_k, n_t)
        ts : tensor (seq,) – time steps
        parametric_iv : optional (batch, n_k, n_t) to add back

        Returns
        -------
        pred_iv : (batch, seq, n_k, n_t) – predicted IV surfaces
        """
        batch, seq = surface_snapshots.shape[:2]

        # Encode last snapshot as initial SDE state
        last_snap = surface_snapshots[:, -1]  # (batch, 1, n_k, n_t)
        z0 = self.encoder(last_snap)           # (batch, latent_dim)
        y0 = self.enc_proj(z0)                 # (batch, state_dim)

        # Integrate SDE
        ys = self.sde(y0, ts)                  # (batch, seq, state_dim)

        # Decode each time step
        decoded = []
        for i in range(seq):
            surf = self.decoder(ys[:, i, :])   # (batch, n_k*n_t)
            decoded.append(surf.view(batch, self.n_k, self.n_t))

        pred_residual = torch.stack(decoded, dim=1)  # (batch, seq, n_k, n_t)

        if parametric_iv is not None:
            pred_residual = pred_residual + parametric_iv.unsqueeze(1)

        return pred_residual
