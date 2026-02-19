"""Neural SDE tensor shape tests."""

import torch
import pytest

from src.neural.neural_sde import NeuralSDE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    return NeuralSDE(state_dim=8, latent_dim=4, hidden_dim=16)


@pytest.fixture
def ts():
    return torch.linspace(0.0, 1.0, 10)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_forward_output_shape(model, ts):
    """NeuralSDE.forward should return (batch, n_steps, state_dim)."""
    batch = 4
    state_dim = model.state_dim
    y0 = torch.randn(batch, state_dim)
    ys = model(y0, ts)
    assert ys.shape == (batch, len(ts), state_dim), (
        f"Expected ({batch}, {len(ts)}, {state_dim}), got {ys.shape}"
    )


def test_forward_first_step_matches_y0(model, ts):
    """First time step of output should equal y0."""
    y0 = torch.randn(2, model.state_dim)
    ys = model(y0, ts)
    assert torch.allclose(ys[:, 0, :], y0, atol=1e-6)


def test_drift_output_shape(model):
    """f(t, y) should have shape (batch, state_dim)."""
    batch = 3
    y = torch.randn(batch, model.state_dim)
    t = torch.tensor(0.5)
    drift = model.f(t, y)
    assert drift.shape == (batch, model.state_dim)


def test_diffusion_output_shape(model):
    """g(t, y) should have shape (batch, state_dim)."""
    batch = 3
    y = torch.randn(batch, model.state_dim)
    t = torch.tensor(0.5)
    diff = model.g(t, y)
    assert diff.shape == (batch, model.state_dim)


def test_diffusion_positive(model):
    """Diffusion g should be strictly positive (Softplus activation)."""
    y = torch.randn(5, model.state_dim)
    t = torch.tensor(0.3)
    diff = model.g(t, y)
    assert torch.all(diff > 0), "Diffusion should be positive"


def test_forward_no_nan(model, ts):
    """Forward pass should not produce NaN or Inf."""
    y0 = torch.randn(4, model.state_dim)
    ys = model(y0, ts)
    assert not torch.isnan(ys).any(), "NaN in NeuralSDE output"
    assert not torch.isinf(ys).any(), "Inf in NeuralSDE output"


@pytest.mark.parametrize("batch,state_dim,n_steps", [
    (1, 4, 5),
    (8, 16, 20),
    (2, 32, 3),
])
def test_forward_various_shapes(batch, state_dim, n_steps):
    """Forward pass should work for various (batch, state_dim, n_steps) combinations."""
    model = NeuralSDE(state_dim=state_dim, latent_dim=8, hidden_dim=16)
    ts = torch.linspace(0, 1, n_steps)
    y0 = torch.randn(batch, state_dim)
    ys = model(y0, ts)
    assert ys.shape == (batch, n_steps, state_dim)


def test_gradient_flows(model, ts):
    """Gradients should flow back through the Euler-Maruyama step."""
    y0 = torch.randn(2, model.state_dim, requires_grad=True)
    ys = model(y0, ts)
    loss = ys.sum()
    loss.backward()
    assert y0.grad is not None
    assert not torch.isnan(y0.grad).any()
