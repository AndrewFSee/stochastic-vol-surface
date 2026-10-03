"""Stress tests for the per-expiry SVI fitter: positive variance, good fits.

Raw SVI attains its minimum at ``a + b*sigma*sqrt(1-rho^2)``.  A fit that dips
below zero in an unquoted wing produces near-zero implied vols on the grid.
The quasi-explicit fitter constrains ``a >= 0`` (and ``b >= 0``), which makes
the minimum non-negative by construction; these tests hold it to that and
check positivity is not bought with a poor fit.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.surface.slices import fit_slices, fit_svi_quasi_explicit, svi_total_variance


def svi_min_variance(a, b, rho, sigma):
    """Analytic minimum of a raw-SVI slice."""
    return a + b * sigma * math.sqrt(max(1.0 - rho ** 2, 0.0))


def _smile(k, atm_var=0.04, skew=-0.05, curv=0.30):
    """A well-behaved convex total-variance smile."""
    return atm_var + skew * k + curv * k ** 2


def _fit(k, w):
    p = fit_svi_quasi_explicit(k, w)
    return svi_total_variance(k, **p), p


# ── The analytic minimum ─────────────────────────────────────────────────


def test_min_variance_matches_numeric_minimum():
    a, b, rho, m, sigma = 0.02, 0.15, -0.6, 0.05, 0.12
    k = np.linspace(-20, 20, 400_001)
    assert svi_min_variance(a, b, rho, sigma) == pytest.approx(
        svi_total_variance(k, a, b, rho, m, sigma).min(), abs=1e-6)


# ── Fitted slices stay non-negative ──────────────────────────────────────


@pytest.mark.parametrize("skew", [0.0, -0.05, -0.20, 0.10])
def test_fit_is_non_negative_for_ordinary_smiles(skew):
    k = np.linspace(-0.4, 0.2, 25)
    _, p = _fit(k, _smile(k, skew=skew))
    assert svi_min_variance(p["a"], p["b"], p["rho"], p["sigma"]) >= -1e-12


def test_fit_stays_non_negative_far_outside_quoted_range():
    """The original failure was in wings no quote covers — check well beyond."""
    k = np.linspace(-0.4, 0.2, 25)
    _, p = _fit(k, _smile(k))
    assert svi_total_variance(np.linspace(-5.0, 5.0, 2001), **p).min() >= -1e-12


def test_fit_on_steep_short_dated_smile():
    """Short-dated slices with a steep put wing triggered the original bug."""
    k = np.linspace(-0.40, 0.20, 30)
    w = 0.0008 + 0.02 * k ** 2 - 0.010 * k
    _, p = _fit(k, w)
    assert svi_total_variance(np.linspace(-3, 3, 1001), **p).min() >= -1e-12


def test_fit_on_asymmetric_wing_data():
    """Data on the put side only, forcing extrapolation to the calls."""
    k = np.linspace(-0.40, -0.05, 20)
    _, p = _fit(k, _smile(k))
    assert svi_total_variance(np.linspace(-2, 2, 1001), **p).min() >= -1e-12


def test_fit_with_noise_stays_non_negative():
    rng = np.random.default_rng(0)
    k = np.linspace(-0.4, 0.2, 40)
    w = np.maximum(_smile(k) * (1 + 0.05 * rng.standard_normal(k.size)), 1e-6)
    _, p = _fit(k, w)
    assert svi_min_variance(p["a"], p["b"], p["rho"], p["sigma"]) >= -1e-12


# ── The fit is still a good fit ──────────────────────────────────────────


def test_fit_still_tracks_the_data():
    """Positivity must not be bought with a badly-fitting slice."""
    k = np.linspace(-0.4, 0.2, 25)
    w = _smile(k)
    w_fit, _ = _fit(k, w)
    assert float(np.sqrt(np.mean((w_fit - w) ** 2))) < 0.01 * w.mean()


def test_atm_level_is_preserved():
    k = np.linspace(-0.4, 0.2, 25)
    w = _smile(k)
    w_fit, _ = _fit(k, w)
    atm = int(np.argmin(np.abs(k)))
    assert w_fit[atm] == pytest.approx(w[atm], rel=0.02)


# ── Chain-level behaviour ────────────────────────────────────────────────


def test_chain_with_no_prices_is_usable_when_iv_is_quoted():
    """A session with blank bid-ask but a quoted IV must not be discarded.

    Some historical sources publish a settled IV with zero bid-ask for a whole
    day.  The price columns are uninformative there, not the rows bad.
    """
    rows = []
    for T in (0.0833, 0.25, 1.0):
        for k in np.linspace(-0.35, 0.15, 15):
            for opt in ("call", "put"):
                rows.append({
                    "strike": 100.0 * np.exp(k), "T": T, "option_type": opt,
                    "bid": 0.0, "ask": 0.0, "mid": 0.0,
                    "implied_volatility_market": 0.20 + 0.3 * k ** 2,
                })
    fits = fit_slices(pd.DataFrame(rows), 100.0, 0.01)
    assert len(fits) == 3
    assert all(f.forward_source == "carry" for f in fits)


def test_built_grid_has_no_floored_cells():
    """End-to-end: a built grid must contain no near-zero implied vols."""
    from datetime import date

    from src.surface.surface import VolSurface
    from tests.synthetic_chains import make_chain

    vs = VolSurface.from_chain(make_chain(), ticker="T", as_of=date(2026, 1, 6),
                               spot=100.0, r=0.04)
    assert np.isfinite(vs.iv_grid).all()
    assert vs.iv_grid.min() > 0.05
