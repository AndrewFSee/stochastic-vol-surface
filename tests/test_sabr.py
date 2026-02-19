"""SABR formula properties and calibration recovery tests."""

import math
import numpy as np
import pytest

from src.models.sabr import sabr_vol, calibrate_sabr


# ── sabr_vol smoke tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("F,K", [(100, 90), (100, 100), (100, 110)])
def test_sabr_vol_positive(F, K):
    """sabr_vol should return a positive implied vol."""
    vol = sabr_vol(F=F, K=K, T=0.5, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    assert math.isfinite(vol), f"sabr_vol not finite for F={F}, K={K}"
    assert vol > 0, f"sabr_vol not positive for F={F}, K={K}"


def test_sabr_atm_formula():
    """ATM SABR (F=K) should equal alpha / F^{1-beta} * (1 + correction*T)."""
    F = 100.0
    alpha, beta, rho, nu, T = 0.2, 0.5, -0.3, 0.4, 1.0
    vol = sabr_vol(F, F, T, alpha, beta, rho, nu)
    # Rough check: vol ≈ alpha / F^{1-beta} near-ATM
    atm_approx = alpha / F ** (1 - beta)
    assert abs(vol - atm_approx) / atm_approx < 0.5, "ATM vol far from approximation"


def test_sabr_vol_nan_for_negative_F():
    """sabr_vol should return NaN for non-positive F."""
    vol = sabr_vol(F=-100, K=100, T=0.5, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    assert math.isnan(vol)


def test_sabr_vol_smile_shape():
    """SABR smile should be roughly symmetric / skewed by rho."""
    F = 100.0
    T, alpha, beta, rho, nu = 0.5, 0.2, 0.5, -0.5, 0.4
    strikes = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
    vols = np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    # With negative rho, put skew: lower strikes should have higher vol
    assert vols[0] > vols[-1], "Expected negative skew (rho < 0)"
    # All vols should be positive
    assert np.all(vols > 0)


# ── calibrate_sabr tests ──────────────────────────────────────────────────────

def test_calibrate_sabr_returns_dict():
    """calibrate_sabr should return a dict with expected keys."""
    F = 100.0
    T = 0.5
    strikes = np.linspace(85, 115, 7)
    alpha_true, beta, rho_true, nu_true = 0.25, 0.5, -0.4, 0.5
    market_vols = np.array([sabr_vol(F, K, T, alpha_true, beta, rho_true, nu_true)
                             for K in strikes])

    result = calibrate_sabr(F, strikes, market_vols, T, beta=beta)
    for key in ("alpha", "beta", "rho", "nu", "rmse"):
        assert key in result, f"Missing key: {key}"


def test_calibrate_sabr_low_rmse():
    """Calibration should fit synthetic data with low RMSE."""
    F = 100.0
    T = 0.5
    strikes = np.linspace(85, 115, 9)
    alpha_true, beta, rho_true, nu_true = 0.20, 0.5, -0.3, 0.40
    market_vols = np.array([sabr_vol(F, K, T, alpha_true, beta, rho_true, nu_true)
                             for K in strikes])

    result = calibrate_sabr(F, strikes, market_vols, T, beta=beta)
    assert result["rmse"] < 1e-3, f"RMSE too large: {result['rmse']}"


def test_calibrate_sabr_parameter_bounds():
    """Calibrated parameters should be within physically sensible bounds."""
    F = 100.0
    T = 1.0
    strikes = np.linspace(80, 120, 9)
    market_vols = np.full(len(strikes), 0.20)

    result = calibrate_sabr(F, strikes, market_vols, T)
    assert result["alpha"] > 0, "alpha must be positive"
    assert result["nu"] >= 0, "nu must be non-negative"
    assert -1 < result["rho"] < 1, "rho must be in (-1, 1)"
    assert result["beta"] == 0.5  # fixed beta
