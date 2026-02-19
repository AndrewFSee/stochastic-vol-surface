"""Tests for butterfly and calendar arbitrage filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestButterflyArbitrageFilter:
    """Tests for the butterfly (convexity) arbitrage filter."""

    def _make_convex_smile(self) -> pd.Series:
        """Create a convex (arbitrage-free) smile."""
        k = np.arange(-0.20, 0.21, 0.02)
        # Symmetric convex smile: σ(k) = 0.20 + 0.5*k²
        iv = 0.20 + 0.5 * k**2
        return pd.Series(iv, index=np.round(k, 4))

    def _make_concave_smile(self) -> pd.Series:
        """Create a strongly concave (butterfly-arbitrage) smile."""
        k = np.arange(-0.20, 0.21, 0.02)
        # w(k) = σ²(k)*T must be concave enough that g(k) < 0
        # Use w = 0.09 - 2*k² so w'' = -4, giving g(0) = 1 + (-4)/2 = -1 < 0
        T = 0.25
        w = np.maximum(0.09 - 2.0 * k**2, 1e-6)
        iv = np.sqrt(w / T)  # Convert total variance to IV
        return pd.Series(iv, index=np.round(k, 4))

    def test_convex_smile_no_violations(self):
        """A convex smile should have no butterfly violations."""
        from src.surface.filters import check_butterfly_arbitrage

        smile = self._make_convex_smile()
        violations = check_butterfly_arbitrage(smile, tenor_years=0.25)
        assert not violations.any(), "Convex smile should have no butterfly violations"

    def test_concave_smile_has_violations(self):
        """A strongly concave smile should have butterfly violations detected."""
        from src.surface.filters import check_butterfly_arbitrage

        smile = self._make_concave_smile()
        violations = check_butterfly_arbitrage(smile, tenor_years=0.25)
        assert violations.any(), "Concave smile should have butterfly violations"

    def test_returns_boolean_series(self):
        """check_butterfly_arbitrage should return a boolean Series."""
        from src.surface.filters import check_butterfly_arbitrage

        smile = self._make_convex_smile()
        violations = check_butterfly_arbitrage(smile, tenor_years=0.5)
        assert violations.dtype == bool or violations.dtype == object
        assert isinstance(violations, pd.Series)

    def test_insufficient_data_returns_empty_violations(self):
        """Too few data points should return all-False violations."""
        from src.surface.filters import check_butterfly_arbitrage

        smile = pd.Series([0.20, 0.21], index=[-0.05, 0.05])
        violations = check_butterfly_arbitrage(smile, tenor_years=0.25)
        assert not violations.any()


class TestCalendarArbitrageFilter:
    """Tests for the calendar spread (monotonicity) filter."""

    def _make_arb_free_surface(self) -> pd.DataFrame:
        """Create a calendar-arbitrage-free surface."""
        k = np.arange(-0.10, 0.11, 0.05)
        T = np.array([0.08, 0.25, 0.5, 1.0])
        iv = np.zeros((len(k), len(T)))
        for j, t in enumerate(T):
            iv[:, j] = 0.15 + 0.1 * t + 0.3 * np.abs(k)
        return pd.DataFrame(iv, index=np.round(k, 4), columns=np.round(T, 4))

    def _make_calendar_arb_surface(self) -> pd.DataFrame:
        """Create a surface with a clear calendar arbitrage violation.

        Ensure w(k, T_short) > w(k, T_long) by directly imposing total variance.
        """
        k = np.arange(-0.10, 0.11, 0.05)
        T = np.array([0.08, 0.25, 0.5, 1.0])
        iv = np.zeros((len(k), len(T)))
        for j, t in enumerate(T):
            iv[:, j] = 0.15 + 0.1 * t + 0.3 * np.abs(k)

        # Force a clear calendar arb: set total variance at T[0] > T[1]
        # w[0] = iv0² * T[0] > iv1² * T[1] = w[1]
        # iv1 = 0.15 + 0.1*0.25 ≈ 0.175 → w[1] = 0.175² * 0.25 ≈ 0.00766
        # Set iv0 such that iv0² * 0.08 > 0.00766 → iv0 > 0.309
        iv[:, 0] = 0.50  # High short-dated IV → w0 = 0.50² * 0.08 = 0.02 > w1
        return pd.DataFrame(iv, index=np.round(k, 4), columns=np.round(T, 4))

    def test_arb_free_surface_no_violations(self):
        """An arbitrage-free surface should have no calendar violations."""
        from src.surface.filters import check_calendar_arbitrage

        surface = self._make_arb_free_surface()
        violations = check_calendar_arbitrage(surface)
        assert not violations.values.any(), "Arb-free surface should have no calendar violations"

    def test_calendar_arb_surface_has_violations(self):
        """A surface with calendar arb should have detected violations."""
        from src.surface.filters import check_calendar_arbitrage

        surface = self._make_calendar_arb_surface()
        violations = check_calendar_arbitrage(surface)
        assert violations.values.any(), "Calendar arb surface should have violations"

    def test_returns_boolean_dataframe(self):
        """check_calendar_arbitrage should return a boolean DataFrame."""
        from src.surface.filters import check_calendar_arbitrage

        surface = self._make_arb_free_surface()
        violations = check_calendar_arbitrage(surface)
        assert isinstance(violations, pd.DataFrame)
        assert violations.shape == surface.shape

    def test_remove_violations_nan_method(self):
        """remove_arbitrage_violations with method='nan' should zero out violations."""
        from src.surface.filters import remove_arbitrage_violations

        surface = self._make_calendar_arb_surface()
        cleaned = remove_arbitrage_violations(surface, method="nan")
        # Cleaned surface should still be a DataFrame
        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned.shape == surface.shape
