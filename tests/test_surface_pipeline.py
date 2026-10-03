"""Tests for the surface construction pipeline (grid_builder → filters → interpolation → VolSurface).

Covers:
- Config loading from surface_grid.yaml
- Forward-moneyness calculation (K/F vs K/S)
- build_surface_grid with synthetic + market IV
- Butterfly filter smoke test
- SVI + RBF interpolation to grid
- VolSurface.from_chain() end-to-end
- VolSurface save / load round-trip
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_chain() -> pd.DataFrame:
    """Minimal synthetic options chain for unit testing.

    Uses artificial mid prices and pre-computed market IV so the tests
    don't depend on the Newton solver converging on toy data.
    """
    spot = 100.0
    r = 0.05
    rows = []
    for T in [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        F = spot * np.exp(r * T)
        for m in np.linspace(-0.30, 0.15, 30):
            K = F * np.exp(m)
            # Flat vol surface at 20% + simple skew
            iv = 0.20 + 0.10 * m - 0.05 * m**2
            rows.append(
                dict(
                    strike=K,
                    T=T,
                    option_type="call" if K >= F else "put",
                    mid=5.0,  # not used when implied_volatility_market present
                    implied_volatility_market=iv,
                    open_interest=500,
                )
            )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────

def test_load_grid_config():
    from src.surface.grid_builder import load_grid_config

    cfg = load_grid_config()
    assert "grid" in cfg
    assert cfg["grid"]["log_moneyness"]["n_points"] == 25


def test_default_grids():
    from src.surface.grid_builder import default_k_grid, default_t_grid

    k = default_k_grid()
    t = default_t_grid()
    assert len(k) == 25
    assert len(t) == 8
    assert k[0] == pytest.approx(-0.40, abs=0.01)
    assert t[-1] == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────────────────
# build_surface_grid
# ──────────────────────────────────────────────────────────────────────────

def test_build_surface_grid_forward_moneyness(synthetic_chain):
    from src.surface.grid_builder import build_surface_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0, r=0.05, use_forward=True)
    assert "log_moneyness" in scatter.columns
    assert "forward" in scatter.columns
    # All forwards should be > spot (rates > 0)
    assert (scatter["forward"] > 100.0).all()


def test_build_surface_grid_legacy_moneyness(synthetic_chain):
    from src.surface.grid_builder import build_surface_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0, r=0.05, use_forward=False)
    # With use_forward=False, forward == spot
    assert (scatter["forward"] == 100.0).all()


def test_build_surface_grid_uses_market_iv(synthetic_chain):
    from src.surface.grid_builder import build_surface_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0)
    # Market IV should be passed through directly
    assert scatter["implied_volatility"].min() > 0
    assert scatter["implied_volatility"].max() < 1.0


# ──────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────

def test_butterfly_filter_preserves_clean_data(synthetic_chain):
    from src.surface.filters import apply_arbitrage_filters
    from src.surface.grid_builder import build_surface_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0)
    filtered = apply_arbitrage_filters(scatter)
    # Our synthetic data is well-behaved; expect few or no removals
    assert len(filtered) >= len(scatter) * 0.8


def test_calendar_filter_grid():
    from src.surface.filters import apply_calendar_filter_grid

    k_grid = np.linspace(-0.3, 0.1, 10)
    t_grid = np.array([0.25, 0.5, 1.0])
    # Deliberately create a calendar violation: T=0.5 > T=1.0 at one point
    iv = np.full((10, 3), 0.20)
    iv[5, 1] = 0.35  # inflated vol at T=0.5 for k_grid[5]
    iv_clean = apply_calendar_filter_grid(k_grid, t_grid, iv)
    # Total variance should now be non-decreasing at k_grid[5]
    for j in range(10):
        tv = iv_clean[j, :] ** 2 * t_grid
        assert np.all(np.diff(tv) >= -1e-10), f"Calendar violation at k index {j}"


# ──────────────────────────────────────────────────────────────────────────
# Interpolation to grid
# ──────────────────────────────────────────────────────────────────────────

def test_interpolate_to_grid_svi(synthetic_chain):
    from src.surface.grid_builder import build_surface_grid, interpolate_to_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0)
    k, t, iv = interpolate_to_grid(scatter, method="svi")
    assert iv.shape == (len(k), len(t))
    # Should have reasonable IV values (non-NaN at interior points)
    mid_k = len(k) // 2
    assert 0.05 < iv[mid_k, 2] < 1.0  # interior should be well-defined


def test_interpolate_to_grid_rbf(synthetic_chain):
    from src.surface.grid_builder import build_surface_grid, interpolate_to_grid

    scatter = build_surface_grid(synthetic_chain, spot=100.0)
    k, t, iv = interpolate_to_grid(scatter, method="rbf")
    assert iv.shape == (len(k), len(t))
    assert np.all(np.isfinite(iv))


# ──────────────────────────────────────────────────────────────────────────
# VolSurface
# ──────────────────────────────────────────────────────────────────────────

def test_vol_surface_from_chain(synthetic_chain):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(
        synthetic_chain,
        ticker="TEST",
        as_of=date(2025, 1, 1),
        spot=100.0,
        r=0.05,
        method="rbf",  # faster for tests
    )
    assert vs.ticker == "TEST"
    assert vs.iv_grid.shape == (25, 8)
    # ATM vol should be close to 0.20 (our synthetic flat level)
    atm = vs.atm_vol(0.5)
    assert 0.10 < atm < 0.40


def test_vol_surface_query(synthetic_chain):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(
        synthetic_chain, ticker="T", as_of=date(2025, 1, 1), spot=100.0,
        method="rbf",
    )
    # Scalar query
    v = vs.iv(log_moneyness=0.0, tenor=0.25)
    assert np.isfinite(v)
    # Array query
    varr = vs.iv(np.array([0.0, 0.05]), np.array([0.25, 0.5]))
    assert varr.shape == (2,)


def test_vol_surface_skew(synthetic_chain):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(
        synthetic_chain, ticker="T", as_of=date(2025, 1, 1), spot=100.0,
        method="rbf",
    )
    s = vs.skew(0.5)
    assert np.isfinite(s)


def test_vol_surface_term_structure(synthetic_chain):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(
        synthetic_chain, ticker="T", as_of=date(2025, 1, 1), spot=100.0,
        method="rbf",
    )
    ts = vs.term_structure()
    assert len(ts) == len(vs.t_grid)
    assert np.all(np.isfinite(ts))


def test_vol_surface_save_load_roundtrip(synthetic_chain):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(
        synthetic_chain, ticker="SPY", as_of=date(2025, 6, 15), spot=100.0,
        r=0.04, method="rbf",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "surface.parquet"
        vs.save(path)
        assert path.exists()

        vs2 = VolSurface.load(path)
        assert vs2.ticker == "SPY"
        assert vs2.as_of == date(2025, 6, 15)
        assert vs2.spot == pytest.approx(100.0)
        assert vs2.r == pytest.approx(0.04)
        np.testing.assert_allclose(vs2.k_grid, vs.k_grid, atol=1e-10)
        np.testing.assert_allclose(vs2.t_grid, vs.t_grid, atol=1e-10)
        np.testing.assert_allclose(vs2.iv_grid, vs.iv_grid, atol=1e-10)


def test_vol_surface_to_from_dataframe():
    from src.surface.surface import VolSurface

    k = np.linspace(-0.2, 0.1, 5)
    t = np.array([0.25, 0.5, 1.0])
    iv = np.random.uniform(0.15, 0.30, (5, 3))

    vs = VolSurface(
        ticker="X", as_of=date(2025, 1, 1),
        k_grid=k, t_grid=t, iv_grid=iv, spot=100.0,
    )
    df = vs.to_dataframe()
    assert len(df) == 15

    vs2 = VolSurface.from_dataframe(df, ticker="X", as_of=date(2025, 1, 1),
                                     k_grid=k, t_grid=t, spot=100.0)
    np.testing.assert_allclose(vs2.iv_grid, iv, atol=1e-10)
