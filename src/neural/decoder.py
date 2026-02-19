"""Latent state decoder: latent vector → predicted IV surface."""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceDecoder(nn.Module):
    """Decode a latent vector into a flattened IV surface.

    Input  : (batch, latent_dim)
    Output : (batch, n_k * n_t)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        n_k: int = 25,
        n_t: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_k = n_k
        self.n_t = n_t
        out_dim = n_k * n_t

        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        # Softplus to ensure positive IVs
        layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (batch, latent_dim)  →  (batch, n_k * n_t)."""
        return self.net(z)
