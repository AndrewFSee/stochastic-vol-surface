"""
Full neural vol model: Encoder → Neural SDE → Decoder.

Learns the residuals between the market IV surface and the best-fit
parametric model.  Given a sequence of surface snapshots, the model:

1. Encodes each snapshot to a latent state z_t.
2. Evolves z_t forward in time via the Neural SDE.
3. Decodes the predicted z_{t+1} back to a surface.

The prediction target is:
    residual(k, T) = σ_market(k, T) - σ_parametric(k, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.neural.encoder import SurfaceEncoder
from src.neural.decoder import SurfaceDecoder
from src.neural.neural_sde import NeuralSDE


class NeuralVolModel(nn.Module):
    """End-to-end neural vol model combining encoder, SDE, and decoder.

    Parameters
    ----------
    n_moneyness:
        Number of log-moneyness grid points.
    n_tenors:
        Number of tenor grid points.
    latent_dim:
        Latent state dimensionality.
    encoder_channels:
        Conv2D channel progression for the encoder.
    decoder_hidden:
        MLP hidden sizes for the decoder.
    drift_hidden, diffusion_hidden:
        MLP hidden sizes for the Neural SDE drift / diffusion.
    """

    def __init__(
        self,
        n_moneyness: int = 41,
        n_tenors: int = 13,
        latent_dim: int = 32,
        encoder_channels: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        drift_hidden: list[int] | None = None,
        diffusion_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SurfaceEncoder(
            n_moneyness=n_moneyness,
            n_tenors=n_tenors,
            latent_dim=latent_dim,
            channels=encoder_channels or [16, 32, 64],
        )
        self.sde = NeuralSDE(
            latent_dim=latent_dim,
            drift_hidden=drift_hidden or [64, 64],
            diffusion_hidden=diffusion_hidden or [64, 64],
        )
        self.decoder = SurfaceDecoder(
            latent_dim=latent_dim,
            n_moneyness=n_moneyness,
            n_tenors=n_tenors,
            hidden_dims=decoder_hidden or [128, 256, 128],
        )
        self.latent_dim = latent_dim

    def encode(self, surface: torch.Tensor) -> torch.Tensor:
        """Encode a batch of surface snapshots to latent states.

        Parameters
        ----------
        surface:
            Shape (batch, n_moneyness, n_tenors) — NaN replaced with 0.
        """
        x = surface.unsqueeze(1)  # Add channel dim
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent states to surface predictions."""
        return self.decoder(z)

    def forward(
        self,
        surface_t: torch.Tensor,
        dt: float = 1.0 / 252.0,
    ) -> torch.Tensor:
        """One-step ahead surface prediction.

        Parameters
        ----------
        surface_t:
            Current surface snapshot, shape (batch, n_moneyness, n_tenors).
        dt:
            Time step in years (default 1 trading day = 1/252).

        Returns
        -------
        torch.Tensor
            Predicted next-day surface, shape (batch, n_moneyness, n_tenors).
        """
        z0 = self.encode(surface_t)
        ts = torch.tensor([0.0, dt], device=z0.device, dtype=z0.dtype)
        z_traj = self.sde(z0, ts)  # (2, batch, latent_dim)
        z1 = z_traj[-1]            # (batch, latent_dim)
        return self.decode(z1)
