"""
Surface snapshot encoder: Conv2D-based encoder mapping a vol surface grid
to a compact latent state vector.

Architecture: Conv2D layers → flatten → linear → latent vector.
Input shape: (batch, 1, n_moneyness, n_tenors).
Output shape: (batch, latent_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceEncoder(nn.Module):
    """Encodes a 2-D vol surface grid into a latent state vector.

    Parameters
    ----------
    n_moneyness:
        Number of log-moneyness grid points.
    n_tenors:
        Number of tenor grid points.
    latent_dim:
        Dimensionality of the output latent vector.
    channels:
        Number of output channels per Conv2D layer, e.g. [16, 32, 64].
    """

    def __init__(
        self,
        n_moneyness: int = 41,
        n_tenors: int = 13,
        latent_dim: int = 32,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32, 64]
        self.latent_dim = latent_dim

        conv_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        # Compute flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_moneyness, n_tenors)
            flat_size = self.conv(dummy).numel()

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a surface batch.

        Parameters
        ----------
        x:
            Input tensor, shape (batch, 1, n_moneyness, n_tenors).
            NaN cells should be replaced with 0 before passing.

        Returns
        -------
        torch.Tensor
            Latent state vectors, shape (batch, latent_dim).
        """
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)
