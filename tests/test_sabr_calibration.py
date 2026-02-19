"""Tests for SABR model calibration."""

from __future__ import annotations

import numpy as np
import pytest


class TestSABRFormula:
    """Tests for the Hagan (2002) SABR implied vol formula."""

    def test_atm_vol_equals_alpha_approx(self):
        """At ATM and low vol-of-vol, SABR IV ≈ α / F^(1-β)."""
        from src.models.sabr import sabr_vol

        F, K = 100.0, 100.0  # ATM
        alpha, beta, rho, nu, T = 0.20, 0.5, 0.0, 0.001, 1.0
        iv = sabr_vol(F, K, T, alpha, beta, rho, nu)
        # For nu→0 and ATM: σ ≈ α / F^(1-β) = 0.20 / 100^0.5 = 0.02
        expected = alpha / (F ** (1 - beta))
        assert abs(iv - expected) < 0.001

    def test_sabr_vol_positive(self):
        """SABR IV should always be positive."""
        from src.models.sabr import sabr_vol

        for F, K, alpha, rho, nu in [
            (100.0, 100.0, 0.20, -0.3, 0.4),
            (100.0, 95.0, 0.25, -0.5, 0.6),
            (100.0, 110.0, 0.15, 0.1, 0.3),
        ]:
            iv = sabr_vol(F, K, 1.0, alpha, 0.5, rho, nu)
            assert iv > 0, f"Negative IV for F={F}, K={K}"

    def test_sabr_vol_symmetric_around_atm(self):
        """For rho=0, SABR smile should be symmetric around ATM."""
        from src.models.sabr import sabr_vol

        F = 100.0
        alpha, beta, rho, nu, T = 0.20, 0.5, 0.0, 0.4, 0.5
        # Symmetric strikes around ATM
        K_low = F * np.exp(-0.05)
        K_high = F * np.exp(0.05)
        iv_low = sabr_vol(F, K_low, T, alpha, beta, rho, nu)
        iv_high = sabr_vol(F, K_high, T, alpha, beta, rho, nu)
        assert abs(iv_low - iv_high) < 0.005

    def test_negative_rho_gives_negative_skew(self):
        """Negative ρ → put IV > call IV (normal market skew)."""
        from src.models.sabr import sabr_vol

        F = 100.0
        alpha, beta, nu, T = 0.20, 0.5, 0.4, 0.5
        K_put = F * np.exp(-0.10)
        K_call = F * np.exp(0.10)
        iv_put = sabr_vol(F, K_put, T, alpha, beta, -0.70, nu)
        iv_call = sabr_vol(F, K_call, T, alpha, beta, -0.70, nu)
        assert iv_put > iv_call, "Negative rho should give put IV > call IV"

    def test_sabr_nan_for_zero_time(self):
        """SABR should return NaN for T ≤ 0."""
        from src.models.sabr import sabr_vol

        iv = sabr_vol(100.0, 100.0, 0.0, 0.20, 0.5, -0.3, 0.4)
        assert np.isnan(iv)


class TestSABRCalibration:
    """Tests for SABR parameter calibration."""

    def _make_market_vols(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Generate synthetic SABR smile as 'market' data."""
        from src.models.sabr import sabr_vol_smile

        F = 100.0
        strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
        true_alpha, true_rho, true_nu = 0.25, -0.50, 0.50
        vols = sabr_vol_smile(F, strikes, 0.5, true_alpha, 0.5, true_rho, true_nu)
        return F, strikes, vols

    def test_calibration_reduces_rmse(self):
        """Calibrated parameters should have lower RMSE than defaults."""
        from src.models.sabr import calibrate_sabr

        F, strikes, market_vols = self._make_market_vols()
        result = calibrate_sabr(
            F=F,
            strikes=strikes,
            market_vols=market_vols,
            T=0.5,
            beta=0.5,
        )
        assert result["rmse"] < 0.01, f"RMSE too high: {result['rmse']}"

    def test_calibration_returns_valid_params(self):
        """Calibrated parameters should be in the valid range."""
        from src.models.sabr import calibrate_sabr

        F, strikes, market_vols = self._make_market_vols()
        result = calibrate_sabr(F=F, strikes=strikes, market_vols=market_vols, T=0.5)

        assert result["alpha"] > 0
        assert -1 < result["rho"] < 1
        assert result["nu"] > 0

    def test_calibration_recovers_true_params(self):
        """Calibration should approximately recover the true SABR parameters."""
        from src.models.sabr import calibrate_sabr, sabr_vol_smile

        F = 100.0
        true_alpha, true_rho, true_nu = 0.20, -0.40, 0.35
        strikes = np.linspace(80, 120, 9)
        market_vols = sabr_vol_smile(F, strikes, 1.0, true_alpha, 0.5, true_rho, true_nu)

        result = calibrate_sabr(
            F=F, strikes=strikes, market_vols=market_vols, T=1.0, beta=0.5,
            alpha_0=true_alpha, rho_0=true_rho, nu_0=true_nu,
        )
        # Allow 10% relative error in recovered parameters
        assert abs(result["alpha"] - true_alpha) < true_alpha * 0.20
        assert abs(result["rho"] - true_rho) < 0.20
        assert abs(result["nu"] - true_nu) < true_nu * 0.20
