"""Tests for VolSurface: grid config, construction, queries and persistence.

Construction details (forwards, IVs, SVI fits, interpolation) are covered in
``test_surface_slices.py``; this file covers the surface object around them.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.synthetic_chains import make_chain, smile_iv


@pytest.fixture(scope="module")
def surface():
    from src.surface.surface import VolSurface

    return VolSurface.from_chain(make_chain(), ticker="TEST", as_of=date(2025, 1, 1),
                                 spot=100.0, r=0.04)


# ── Grid config ──────────────────────────────────────────────────────────


def test_load_grid_config():
    from src.surface.grid import load_grid_config

    cfg = load_grid_config()
    assert cfg["grid"]["log_moneyness"]["n_points"] == 25


def test_default_grids():
    from src.surface.grid import default_k_grid, default_t_grid

    k, t = default_k_grid(), default_t_grid()
    assert len(k) == 25 and len(t) == 8
    assert k[0] == pytest.approx(-0.40, abs=0.01)
    assert t[-1] == pytest.approx(2.0)


def test_calendar_filter_grid():
    from src.surface.filters import apply_calendar_filter_grid

    k_grid = np.linspace(-0.3, 0.1, 10)
    t_grid = np.array([0.25, 0.5, 1.0])
    iv = np.full((10, 3), 0.20)
    iv[5, 1] = 0.35  # total variance at T=0.5 now exceeds T=1.0 at k_grid[5]
    iv_clean = apply_calendar_filter_grid(k_grid, t_grid, iv)
    for j in range(10):
        assert np.all(np.diff(iv_clean[j, :] ** 2 * t_grid) >= -1e-10)


# ── Construction ─────────────────────────────────────────────────────────


def test_from_chain_shape_and_level(surface):
    assert surface.ticker == "TEST"
    assert surface.iv_grid.shape == (25, 8)
    assert surface.atm_vol(0.5) == pytest.approx(smile_iv(0.0, 0.5), abs=0.005)


def test_from_chain_custom_grid():
    from src.surface.surface import VolSurface

    k, t = np.linspace(-0.2, 0.1, 7), np.array([0.25, 0.5, 1.0])
    vs = VolSurface.from_chain(make_chain(), ticker="T", as_of=date(2025, 1, 1),
                               spot=100.0, k_grid=k, t_grid=t)
    assert vs.iv_grid.shape == (7, 3)


def test_from_chain_needs_two_expiries():
    from src.surface.surface import VolSurface

    with pytest.raises(ValueError, match="expiries"):
        VolSurface.from_chain(make_chain(tenors=(0.25,)), ticker="T",
                              as_of=date(2025, 1, 1), spot=100.0)


# ── Queries ──────────────────────────────────────────────────────────────


def test_query_scalar_and_array(surface):
    assert np.isfinite(surface.iv(log_moneyness=0.0, tenor=0.25))
    assert surface.iv(np.array([0.0, 0.05]), np.array([0.25, 0.5])).shape == (2,)


def test_skew_is_positive_under_put_skew(surface):
    assert surface.skew(0.5) > 0          # put minus call


def test_term_structure(surface):
    ts = surface.term_structure()
    assert len(ts) == len(surface.t_grid) and np.all(np.isfinite(ts))


# ── Persistence ──────────────────────────────────────────────────────────


def test_save_load_roundtrip(surface, tmp_path):
    from src.surface.surface import VolSurface

    vs2 = VolSurface.load(surface.save(tmp_path / "surface.parquet"))
    assert vs2.ticker == "TEST" and vs2.as_of == date(2025, 1, 1)
    assert vs2.spot == pytest.approx(100.0) and vs2.r == pytest.approx(0.04)
    np.testing.assert_allclose(vs2.k_grid, surface.k_grid, atol=1e-10)
    np.testing.assert_allclose(vs2.t_grid, surface.t_grid, atol=1e-10)
    np.testing.assert_allclose(vs2.iv_grid, surface.iv_grid, atol=1e-10)


def test_to_from_dataframe():
    from src.surface.surface import VolSurface

    k = np.linspace(-0.2, 0.1, 5)
    t = np.array([0.25, 0.5, 1.0])
    iv = np.random.default_rng(0).uniform(0.15, 0.30, (5, 3))
    vs = VolSurface(ticker="X", as_of=date(2025, 1, 1), k_grid=k, t_grid=t,
                    iv_grid=iv, spot=100.0)
    df = vs.to_dataframe()
    assert len(df) == 15
    vs2 = VolSurface.from_dataframe(df, ticker="X", as_of=date(2025, 1, 1),
                                    k_grid=k, t_grid=t, spot=100.0)
    np.testing.assert_allclose(vs2.iv_grid, iv, atol=1e-10)
