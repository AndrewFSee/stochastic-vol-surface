"""Tests for vol-arb signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_test_surface(atm_vol: float = 0.20, skew_slope: float = 0.5) -> "VolSurface":
    """Create a test VolSurface with configurable skew.

    Uses a negatively-sloped smile (put IV > call IV) when skew_slope > 0.
    The smile is: σ(k, T) = atm_vol - skew_slope * k + 0.01*T
    (i.e., put-wing IV > ATM > call-wing IV for k<0 and k>0 respectively)
    """
    from src.surface.surface import VolSurface

    moneyness = np.arange(-0.20, 0.21, 0.05)
    tenors = np.array([1 / 12, 3 / 12, 6 / 12, 1.0])
    iv = np.zeros((len(moneyness), len(tenors)))
    for i, k in enumerate(moneyness):
        for j, T in enumerate(tenors):
            # Negatively sloped smile: IV decreases as k increases
            # (higher for puts at k<0, lower for calls at k>0)
            iv[i, j] = atm_vol - skew_slope * k + 0.01 * T
    grid = pd.DataFrame(iv, index=np.round(moneyness, 4), columns=np.round(tenors, 6))
    return VolSurface(grid, ticker="TEST", spot=100.0)


class TestSkewSignals:
    """Tests for 25-delta skew signal generation."""

    def test_positive_skew_for_normal_smile(self):
        """A negatively-sloped smile should have positive skew (put IV > call IV)."""
        from src.signals.skew_signals import compute_skew

        surf = _make_test_surface(atm_vol=0.20, skew_slope=0.5)
        skew = compute_skew(surf, tenor_years=0.25)
        assert skew > 0, "Negatively-sloped smile should have positive put-call skew"

    def test_zero_skew_for_flat_smile(self):
        """A perfectly flat smile should have near-zero skew."""
        from src.surface.surface import VolSurface
        from src.signals.skew_signals import compute_skew

        moneyness = np.arange(-0.20, 0.21, 0.05)
        tenors = np.array([0.083, 0.25, 0.5, 1.0])
        # Flat vol surface (same IV everywhere)
        iv = np.full((len(moneyness), len(tenors)), 0.20)
        grid = pd.DataFrame(iv, index=np.round(moneyness, 4), columns=np.round(tenors, 6))
        surf = VolSurface(grid, spot=100.0)

        skew = compute_skew(surf, tenor_years=0.25)
        assert np.isfinite(skew)
        assert abs(skew) < 0.01

    def test_detect_signals_returns_dataframe(self):
        """detect_skew_signals should return a DataFrame with required columns."""
        from src.signals.skew_signals import detect_skew_signals

        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        skew = pd.Series(0.05 + np.random.randn(100) * 0.01, index=dates)
        df = detect_skew_signals(skew, zscore_threshold=1.5, lookback=30)

        assert isinstance(df, pd.DataFrame)
        assert "skew" in df.columns
        assert "zscore" in df.columns
        assert "signal" in df.columns

    def test_signal_values_in_valid_range(self):
        """Signal values should be -1, 0, or +1."""
        from src.signals.skew_signals import detect_skew_signals

        dates = pd.date_range("2024-01-01", periods=200)
        np.random.seed(0)
        skew = pd.Series(np.random.randn(200) * 0.02 + 0.05, index=dates)
        df = detect_skew_signals(skew)

        assert set(df["signal"].unique()).issubset({-1, 0, 1})


class TestTermStructureSignals:
    """Tests for term structure signal detection."""

    def test_upward_sloping_positive_slope(self):
        """Upward-sloping term structure should give positive slope."""
        from src.signals.term_structure import compute_term_structure_slope

        surf = _make_test_surface()
        slope = compute_term_structure_slope(surf, short_tenor=1 / 12, long_tenor=1.0)
        assert slope > 0

    def test_inversion_detection_true(self):
        """Inverted surface should be detected as inverted."""
        from src.surface.surface import VolSurface
        from src.signals.term_structure import detect_inversion

        moneyness = np.arange(-0.05, 0.06, 0.05)  # 3 points to allow interpolation
        tenors = np.array([0.02, 0.08, 0.25, 0.5, 1.0])  # Use exact decimal tenors
        # Inverted: short vol higher than long vol across all moneyness levels
        iv = np.zeros((len(moneyness), len(tenors)))
        for i in range(len(moneyness)):
            iv[i, :] = [0.40, 0.35, 0.30, 0.25, 0.20]  # Decreasing IV with tenor
        grid = pd.DataFrame(iv, index=np.round(moneyness, 4), columns=tenors)
        surf = VolSurface(grid, spot=100.0)

        # Use tenors that fall within the grid range
        assert detect_inversion(surf, tenors_years=[0.02, 0.08, 0.25, 0.5, 1.0],
                                 threshold=-0.05)

    def test_inversion_detection_false(self):
        """Normal upward-sloping surface should not be detected as inverted."""
        from src.signals.term_structure import detect_inversion

        surf = _make_test_surface()
        assert not detect_inversion(surf)


class TestRegimeClassifier:
    """Tests for the vol regime classifier."""

    def test_low_regime_below_threshold(self):
        """VIX < 15 should give 'Low' regime."""
        from src.signals.regime_vol import classify_vol_regime

        assert classify_vol_regime(12.0) == "Low"

    def test_normal_regime(self):
        """VIX between 15 and 25 should give 'Normal' regime."""
        from src.signals.regime_vol import classify_vol_regime

        assert classify_vol_regime(20.0) == "Normal"

    def test_high_regime(self):
        """VIX between 25 and 40 should give 'High' regime."""
        from src.signals.regime_vol import classify_vol_regime

        assert classify_vol_regime(30.0) == "High"

    def test_crisis_regime(self):
        """VIX ≥ 40 should give 'Crisis' regime."""
        from src.signals.regime_vol import classify_vol_regime

        assert classify_vol_regime(50.0) == "Crisis"

    def test_position_sizing_decreases_in_crisis(self):
        """Position size should decrease in higher vol regimes."""
        from src.signals.regime_vol import regime_adjusted_position_size

        base = 1.0
        low = regime_adjusted_position_size(base, "Low")
        high = regime_adjusted_position_size(base, "High")
        crisis = regime_adjusted_position_size(base, "Crisis")

        assert low >= high >= crisis
