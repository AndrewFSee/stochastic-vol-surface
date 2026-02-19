"""Tests for butterfly and calendar arbitrage filters."""

import numpy as np
import pytest

from src.surface.filters import (
    butterfly_arbitrage_filter,
    calendar_arbitrage_filter,
    remove_calendar_violations,
)


# ── Butterfly filter ──────────────────────────────────────────────────────────

def _flat_smile(n: int = 20, iv: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """A flat smile – trivially arbitrage-free."""
    strikes = np.linspace(80, 120, n)
    ivs = np.full(n, iv)
    return strikes, ivs


def _convex_smile(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A convex smile – arbitrage-free by construction."""
    strikes = np.linspace(80, 120, n)
    k = np.log(strikes / 100.0)
    ivs = 0.20 + 0.05 * k ** 2  # upward-opening parabola → convex
    return strikes, ivs


def _concave_smile(n: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """A strongly concave total-variance smile – has butterfly arbitrage.

    Constructs w(k) = 0.10 - 2*k^2 (concave in total variance), which gives
    g(k) < 0 near ATM per the Gatheral density formula.
    """
    strikes = np.linspace(90, 110, n)
    k = np.log(strikes / 100.0)
    T = 0.25
    w = np.maximum(0.10 - 2.0 * k ** 2, 1e-4)   # total variance w = sigma^2 * T
    ivs = np.sqrt(w / T)
    return strikes, ivs


def test_butterfly_flat_smile_is_arbitrage_free():
    strikes, ivs = _flat_smile()
    is_free, g = butterfly_arbitrage_filter(strikes, ivs, F=100.0, T=0.25)
    assert is_free, "Flat smile should be butterfly-arbitrage-free"


def test_butterfly_convex_smile_is_arbitrage_free():
    strikes, ivs = _convex_smile()
    is_free, g = butterfly_arbitrage_filter(strikes, ivs, F=100.0, T=0.25)
    assert is_free, "Convex smile should be butterfly-arbitrage-free"


def test_butterfly_concave_smile_has_arbitrage():
    strikes, ivs = _concave_smile()
    is_free, g = butterfly_arbitrage_filter(strikes, ivs, F=100.0, T=0.25)
    # Concave region will have g < 0 (butterfly arbitrage)
    assert not is_free, "Concave smile should have butterfly arbitrage"
    assert np.any(g < 0), "g should be negative somewhere for concave smile"


def test_butterfly_returns_g_array():
    strikes, ivs = _flat_smile()
    is_free, g = butterfly_arbitrage_filter(strikes, ivs, F=100.0, T=0.25)
    assert g.shape == ivs.shape


# ── Calendar filter ───────────────────────────────────────────────────────────

def _monotone_tv() -> dict[float, np.ndarray]:
    """Total variance increasing in tenor – no calendar arbitrage."""
    k_grid = np.linspace(-0.3, 0.2, 15)
    return {
        0.25: 0.20 ** 2 * 0.25 * np.ones(15),
        0.50: 0.21 ** 2 * 0.50 * np.ones(15),
        1.00: 0.22 ** 2 * 1.00 * np.ones(15),
    }


def _non_monotone_tv() -> dict[float, np.ndarray]:
    """Total variance decreasing from 0.25 to 0.50 – calendar arbitrage."""
    k_grid = np.linspace(-0.3, 0.2, 15)
    return {
        0.25: 0.25 ** 2 * 0.25 * np.ones(15),   # higher TV at near tenor
        0.50: 0.10 ** 2 * 0.50 * np.ones(15),   # lower TV at far tenor
        1.00: 0.22 ** 2 * 1.00 * np.ones(15),
    }


def test_calendar_monotone_is_arbitrage_free():
    is_free, violations = calendar_arbitrage_filter(_monotone_tv())
    assert is_free, f"Monotone TV should be calendar-arb-free, got: {violations}"
    assert violations == []


def test_calendar_non_monotone_has_arbitrage():
    is_free, violations = calendar_arbitrage_filter(_non_monotone_tv())
    assert not is_free, "Non-monotone TV should have calendar arbitrage"
    assert len(violations) > 0


def test_remove_calendar_violations_makes_monotone():
    tv = _non_monotone_tv()
    cleaned = remove_calendar_violations(tv)
    is_free, violations = calendar_arbitrage_filter(cleaned)
    assert is_free, f"After cleaning, should be arb-free; violations: {violations}"


def test_calendar_returns_tenor_pairs_in_violations():
    is_free, violations = calendar_arbitrage_filter(_non_monotone_tv())
    for v in violations:
        T1, T2 = v
        assert T1 < T2, "Violation should list (earlier, later) tenors"
