"""Neural SDE: drift + diffusion as MLPs, Euler-Maruyama integration.

Falls back to a pure-PyTorch Euler-Maruyama implementation so torchsde
is an optional dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """Simple MLP with tanh activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralSDE(nn.Module):
    """Neural SDE for latent volatility surface dynamics.

    State dim *state_dim* is the latent vol-surface representation.
    The model takes an extra time input, so the MLP input is (state_dim + 1).

    Forward pass uses Euler-Maruyama:
        Y_{t+dt} = Y_t + f(t, Y_t)*dt + g(t, Y_t)*sqrt(dt)*eps,  eps ~ N(0,I)
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        in_dim = state_dim + 1  # state + scalar time

        # Drift network: f(t, y) → R^{state_dim}
        self._drift_net = _MLP(in_dim, state_dim, hidden_dim, n_layers)
        # Diffusion network: g(t, y) → R^{state_dim}  (diagonal diffusion)
        self._diffusion_net = _MLP(in_dim, state_dim, hidden_dim, n_layers)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: f(t, y), shape (batch, state_dim)."""
        t_expand = t.expand(y.shape[0], 1) if t.dim() == 0 else t.view(-1, 1).expand(y.shape[0], 1)
        return self._drift_net(torch.cat([y, t_expand], dim=-1))

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion (diagonal): g(t, y), shape (batch, state_dim)."""
        t_expand = t.expand(y.shape[0], 1) if t.dim() == 0 else t.view(-1, 1).expand(y.shape[0], 1)
        # Softplus to ensure positive diffusion
        return nn.functional.softplus(self._diffusion_net(torch.cat([y, t_expand], dim=-1)))

    def forward(
        self,
        y0: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """Euler-Maruyama integration.

        Parameters
        ----------
        y0 : torch.Tensor  shape (batch, state_dim) – initial state
        ts : torch.Tensor  shape (n_steps,) – time grid (must be increasing)

        Returns
        -------
        ys : torch.Tensor  shape (batch, n_steps, state_dim)
        """
        batch = y0.shape[0]
        n_steps = len(ts)
        ys = torch.empty(batch, n_steps, self.state_dim, device=y0.device, dtype=y0.dtype)
        ys[:, 0, :] = y0

        for i in range(1, n_steps):
            dt = ts[i] - ts[i - 1]
            t_i = ts[i - 1]
            y_i = ys[:, i - 1, :]
            drift = self.f(t_i, y_i)
            diffusion = self.g(t_i, y_i)
            eps = torch.randn_like(y_i)
            ys[:, i, :] = y_i + drift * dt + diffusion * eps * dt.sqrt()

        return ys
