"""
Neural SDE definition using torchsde.

Defines the drift (μ) and diffusion (σ) networks of the Neural SDE as MLPs.
The SDE models the latent dynamics of the vol surface residuals:

    dZ_t = μ_θ(Z_t, t) dt + σ_φ(Z_t, t) dW_t

where Z_t is a latent state encoding the vol surface and W_t is a Brownian
motion.  The drift and diffusion are parameterised as neural networks.

Reference:
    Kidger, P., Foster, J., Li, X., & Lyons, T. (2021).
    Neural SDEs as Infinite-Dimensional GANs. ICML 2021. arXiv:2102.03657.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """Simple feedforward MLP with Tanh activations."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim] + hidden_dims
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralSDE(nn.Module):
    """Neural SDE with drift and diffusion parameterised as MLPs.

    Compatible with the ``torchsde`` library interface.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent state Z.
    drift_hidden:
        Hidden layer sizes for the drift network.
    diffusion_hidden:
        Hidden layer sizes for the diffusion network.
    noise_type:
        ``"diagonal"`` — each latent dim has its own noise (faster).
    sde_type:
        ``"ito"`` (default) or ``"stratonovich"``.
    """

    noise_type: str
    sde_type: str

    def __init__(
        self,
        latent_dim: int = 32,
        drift_hidden: list[int] | None = None,
        diffusion_hidden: list[int] | None = None,
        noise_type: str = "diagonal",
        sde_type: str = "ito",
    ) -> None:
        super().__init__()
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.latent_dim = latent_dim

        drift_hidden = drift_hidden or [64, 64]
        diffusion_hidden = diffusion_hidden or [64, 64]

        # Input: [Z, t] → latent_dim + 1
        in_dim = latent_dim + 1

        self.drift_net = _MLP(in_dim, drift_hidden, latent_dim)
        self.diffusion_net = _MLP(in_dim, diffusion_hidden, latent_dim)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift function μ(Z_t, t)."""
        # Expand t to match batch dimension
        t_expand = t.expand(y.shape[0], 1) if y.dim() > 1 else t.unsqueeze(0)
        inp = torch.cat([y, t_expand], dim=-1)
        return self.drift_net(inp)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion function σ(Z_t, t) (diagonal noise)."""
        t_expand = t.expand(y.shape[0], 1) if y.dim() > 1 else t.unsqueeze(0)
        inp = torch.cat([y, t_expand], dim=-1)
        # Positive diffusion via softplus
        return nn.functional.softplus(self.diffusion_net(inp))

    def forward(
        self,
        z0: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the SDE from z0 over time grid ts using torchsde.

        Parameters
        ----------
        z0:
            Initial latent state, shape (batch, latent_dim).
        ts:
            Time grid tensor, shape (n_steps,).

        Returns
        -------
        torch.Tensor
            Latent trajectory, shape (n_steps, batch, latent_dim).
        """
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError("torchsde is required: pip install torchsde") from exc

        # torchsde expects y0 shape (batch, state_dim)
        ys = torchsde.sdeint(self, z0, ts, method="euler", dt=ts[1] - ts[0])
        return ys
