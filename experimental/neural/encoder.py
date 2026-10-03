"""Surface snapshot encoder: Conv2D → latent vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceEncoder(nn.Module):
    """Encode a 2-D IV surface snapshot into a latent vector.

    Input  : (batch, 1, n_k, n_t)   – single-channel surface image
    Output : (batch, latent_dim)
    """

    def __init__(
        self,
        n_k: int = 25,
        n_t: int = 8,
        latent_dim: int = 16,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.n_k = n_k
        self.n_t = n_t
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (batch, base_channels*2, 4, 4)
        )

        flat_dim = base_channels * 2 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, 1, n_k, n_t)  →  (batch, latent_dim)."""
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)
