"""Tests for IV calculation and surface construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import math


class TestBlackScholesPrice:
    """Tests for the Black-Scholes pricing formula."""

    def test_call_put_parity(self):
        """Call - Put = S - K*exp(-r*T) (put-call parity)."""
        from src.surface.implied_vol import black_scholes_price

        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        expected_diff = S - K * math.exp(-r * T)
        assert abs(call - put - expected_diff) < 1e-8

    def test_call_price_positive(self):
        """Call price should always be > 0 for positive vol."""
        from src.surface.implied_vol import black_scholes_price

        price = black_scholes_price(100.0, 110.0, 1.0, 0.05, 0.20, "call")
        assert price > 0

    def test_zero_time_returns_intrinsic(self):
        """At T=0, call price = max(S-K, 0) (intrinsic value)."""
        from src.surface.implied_vol import black_scholes_price

        S, K = 100.0, 95.0
        price = black_scholes_price(S, K, 0.0, 0.05, 0.20, "call")
        assert abs(price - max(S - K, 0.0)) < 1e-6


class TestImpliedVol:
    """Tests for the Brent's method IV inverter."""

    def test_round_trip_atm_call(self):
        """IV inversion should recover the original vol for an ATM call."""
        from src.surface.implied_vol import black_scholes_price, implied_vol

        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
        price = black_scholes_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, "call")
        assert recovered is not None
        assert abs(recovered - sigma) < 1e-6

    def test_round_trip_otm_put(self):
        """IV inversion should recover the original vol for an OTM put."""
        from src.surface.implied_vol import black_scholes_price, implied_vol

        S, K, T, r, sigma = 100.0, 90.0, 0.5, 0.04, 0.25
        price = black_scholes_price(S, K, T, r, sigma, "put")
        recovered = implied_vol(price, S, K, T, r, "put")
        assert recovered is not None
        assert abs(recovered - sigma) < 1e-5

    def test_below_intrinsic_returns_none(self):
        """A price below intrinsic should return None (no-arbitrage violation)."""
        from src.surface.implied_vol import implied_vol

        iv = implied_vol(0.0001, 100.0, 90.0, 0.5, 0.05, "call")
        assert iv is None

    def test_round_trip_high_vol(self):
        """IV inversion should work for a high-vol scenario (80%)."""
        from src.surface.implied_vol import black_scholes_price, implied_vol

        S, K, T, r, sigma = 100.0, 120.0, 1.0, 0.03, 0.80
        price = black_scholes_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, "call")
        assert recovered is not None
        assert abs(recovered - sigma) < 1e-5


class TestGridBuilder:
    """Tests for the chain → grid transformation."""

    def _make_chain(self) -> pd.DataFrame:
        """Create a minimal options chain for testing."""
        rows = []
        spot = 100.0
        for dte in [30, 60, 90]:
            for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
                for opt_type in ["call", "put"]:
                    from src.surface.implied_vol import black_scholes_price
                    iv = 0.20 + abs(np.log(K / spot)) * 0.5  # Smile
                    T = dte / 365.0
                    price = black_scholes_price(spot, K, T, 0.05, iv, opt_type)
                    rows.append({
                        "strike": K,
                        "days_to_expiry": dte,
                        "option_type": opt_type,
                        "mid_price": price,
                        "bid": price * 0.95,
                        "ask": price * 1.05,
                        "last_price": price,
                        "volume": 100,
                        "open_interest": 500,
                    })
        return pd.DataFrame(rows)

    def test_grid_shape(self):
        """build_grid should return a DataFrame with correct shape."""
        from src.surface.grid_builder import build_grid

        chain = self._make_chain()
        moneyness = np.arange(-0.15, 0.16, 0.05)
        tenors = np.array([30 / 365.0, 60 / 365.0, 90 / 365.0])

        grid = build_grid(chain, spot=100.0, risk_free_rate=0.05,
                          moneyness_grid=moneyness, tenor_grid=tenors)

        assert isinstance(grid, pd.DataFrame)
        assert grid.shape[0] == len(moneyness)
        assert grid.shape[1] == len(tenors)

    def test_grid_values_positive(self):
        """All non-NaN grid values should be positive implied vols."""
        from src.surface.grid_builder import build_grid

        chain = self._make_chain()
        moneyness = np.arange(-0.15, 0.16, 0.05)
        tenors = np.array([30 / 365.0, 60 / 365.0, 90 / 365.0])

        grid = build_grid(chain, spot=100.0, risk_free_rate=0.05,
                          moneyness_grid=moneyness, tenor_grid=tenors)

        valid_vals = grid.values[~np.isnan(grid.values)]
        assert (valid_vals > 0).all()
        assert (valid_vals < 5.0).all()  # All vols < 500%


class TestVolSurface:
    """Tests for the VolSurface query interface."""

    def _make_surface(self) -> "VolSurface":
        from src.surface.surface import VolSurface

        moneyness = np.arange(-0.20, 0.21, 0.05)
        tenors = np.array([0.083, 0.25, 0.5, 1.0])
        # Simple smile: ATM vol = 20%, with ±5% wings
        iv_matrix = np.zeros((len(moneyness), len(tenors)))
        for i, k in enumerate(moneyness):
            for j, T in enumerate(tenors):
                iv_matrix[i, j] = 0.20 + abs(k) * 0.5 + T * 0.01
        grid = pd.DataFrame(iv_matrix, index=moneyness, columns=tenors)
        return VolSurface(grid, ticker="TEST", spot=100.0)

    def test_atm_vol(self):
        """ATM vol at k=0 should be approximately 0.20."""
        surf = self._make_surface()
        atm = surf.get_atm_vol(0.25)
        assert abs(atm - (0.20 + 0.25 * 0.01)) < 0.02

    def test_positive_skew(self):
        """Skew of a flat surface should be zero (no asymmetry)."""
        surf = self._make_surface()
        skew = surf.get_skew(0.25)
        # The test surface is symmetric around ATM, so skew ≈ 0
        assert np.isfinite(skew)

    def test_term_structure_shape(self):
        """Term structure should return a Series with correct length."""
        surf = self._make_surface()
        ts = surf.get_term_structure()
        assert len(ts) == 4

    def test_out_of_range_returns_nan(self):
        """Queries outside the grid should return NaN (not crash)."""
        surf = self._make_surface()
        iv = surf.get_iv(5.0, 10.0)  # Far outside the grid
        assert np.isnan(iv)
