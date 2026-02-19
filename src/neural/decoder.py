"""
Surface decoder: maps a latent state vector back to a predicted IV surface grid.

Architecture: MLP → reshape → (batch, n_moneyness, n_tenors).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceDecoder(nn.Module):
    """Decodes a latent vector to a predicted vol surface grid.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the input latent vector.
    n_moneyness:
        Number of log-moneyness grid points.
    n_tenors:
        Number of tenor grid points.
    hidden_dims:
        Hidden layer sizes of the MLP, e.g. [128, 256, 128].
    """

    def __init__(
        self,
        latent_dim: int = 32,
        n_moneyness: int = 41,
        n_tenors: int = 13,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 256, 128]
        self.n_moneyness = n_moneyness
        self.n_tenors = n_tenors
        out_dim = n_moneyness * n_tenors

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, out_dim))
        # Softplus ensures positive output (IVs are > 0)
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to surface grids.

        Parameters
        ----------
        z:
            Latent state, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Predicted IV surface, shape (batch, n_moneyness, n_tenors).
        """
        out = self.mlp(z)
        return out.view(-1, self.n_moneyness, self.n_tenors)
