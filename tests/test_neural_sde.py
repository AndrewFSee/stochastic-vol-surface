"""Tests for the Neural SDE model architecture."""

from __future__ import annotations

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


class TestNeuralSDEForwardPass:
    """Tests for the Neural SDE architecture without torchsde."""

    def test_mlp_shapes(self):
        """MLP inside NeuralSDE should produce correct output shapes."""
        import torch
        from src.neural.neural_sde import _MLP

        mlp = _MLP(in_dim=10, hidden_dims=[32, 32], out_dim=5)
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 5)

    def test_drift_diffusion_shapes(self):
        """Drift and diffusion networks should output (batch, latent_dim)."""
        import torch
        from src.neural.neural_sde import NeuralSDE

        latent_dim = 16
        sde = NeuralSDE(latent_dim=latent_dim, drift_hidden=[32], diffusion_hidden=[32])

        batch = 8
        z = torch.randn(batch, latent_dim)
        t = torch.tensor(0.5)

        f_out = sde.f(t, z)
        g_out = sde.g(t, z)

        assert f_out.shape == (batch, latent_dim)
        assert g_out.shape == (batch, latent_dim)

    def test_diffusion_is_positive(self):
        """Diffusion (Softplus-activated) should always be positive."""
        import torch
        from src.neural.neural_sde import NeuralSDE

        sde = NeuralSDE(latent_dim=8)
        z = torch.randn(4, 8)
        t = torch.tensor(0.1)
        g = sde.g(t, z)
        assert (g > 0).all(), "Diffusion must be strictly positive"


class TestSurfaceEncoder:
    """Tests for the surface encoder architecture."""

    def test_output_shape(self):
        """Encoder should output a vector of shape (batch, latent_dim)."""
        import torch
        from src.neural.encoder import SurfaceEncoder

        n_m, n_T, latent_dim = 41, 13, 32
        encoder = SurfaceEncoder(n_moneyness=n_m, n_tenors=n_T, latent_dim=latent_dim)
        x = torch.randn(4, 1, n_m, n_T)
        z = encoder(x)
        assert z.shape == (4, latent_dim)

    def test_small_input(self):
        """Encoder should handle small grid sizes without error."""
        import torch
        from src.neural.encoder import SurfaceEncoder

        encoder = SurfaceEncoder(n_moneyness=8, n_tenors=5, latent_dim=16,
                                  channels=[8, 16])
        x = torch.randn(2, 1, 8, 5)
        z = encoder(x)
        assert z.shape == (2, 16)


class TestSurfaceDecoder:
    """Tests for the surface decoder architecture."""

    def test_output_shape(self):
        """Decoder should output a surface of shape (batch, n_m, n_T)."""
        import torch
        from src.neural.decoder import SurfaceDecoder

        latent_dim, n_m, n_T = 32, 41, 13
        decoder = SurfaceDecoder(latent_dim=latent_dim, n_moneyness=n_m, n_tenors=n_T)
        z = torch.randn(4, latent_dim)
        out = decoder(z)
        assert out.shape == (4, n_m, n_T)

    def test_output_is_positive(self):
        """Decoder output (Softplus-activated) should always be positive."""
        import torch
        from src.neural.decoder import SurfaceDecoder

        decoder = SurfaceDecoder(latent_dim=16, n_moneyness=10, n_tenors=5)
        z = torch.randn(8, 16)
        out = decoder(z)
        assert (out > 0).all(), "Decoder output must be positive (IV > 0)"


class TestVolSurfaceLoss:
    """Tests for the custom vol surface loss functions."""

    def test_weighted_mse_zero_for_equal_tensors(self):
        """Weighted MSE loss should be zero when pred == target."""
        import torch
        from src.neural.losses import weighted_mse_loss

        n_m, n_T = 10, 5
        pred = torch.rand(4, n_m, n_T)
        moneyness = torch.linspace(-0.10, 0.10, n_m)
        loss = weighted_mse_loss(pred, pred.clone(), moneyness)
        assert abs(loss.item()) < 1e-6

    def test_loss_higher_for_atm_errors(self):
        """ATM region errors should contribute more to the loss."""
        import torch
        from src.neural.losses import weighted_mse_loss

        n_m, n_T = 11, 5
        target = torch.zeros(1, n_m, n_T)
        moneyness = torch.linspace(-0.10, 0.10, n_m)

        # Error only at ATM (middle)
        atm_pred = torch.zeros(1, n_m, n_T)
        atm_pred[0, n_m // 2, :] = 0.05

        # Error only at wing
        wing_pred = torch.zeros(1, n_m, n_T)
        wing_pred[0, 0, :] = 0.05

        atm_loss = weighted_mse_loss(atm_pred, target, moneyness, atm_weight=3.0)
        wing_loss = weighted_mse_loss(wing_pred, target, moneyness, atm_weight=3.0)

        assert atm_loss > wing_loss, "ATM error should create higher loss"
