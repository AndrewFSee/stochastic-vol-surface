"""Tests for per-expiry surface construction (src.surface.slices)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear

from src.surface import slices as S
from src.surface.implied_vol import black_implied_vol, black_price
from tests.synthetic_chains import make_chain, smile_iv


# ── Black-76 inversion ───────────────────────────────────────────────────


def test_black_iv_round_trip():
    rng = np.random.default_rng(0)
    n = 2000
    F = 100.0
    K = F * np.exp(rng.uniform(-0.8, 0.5, n))
    T = rng.uniform(2 / 365, 2.5, n)
    sig = rng.uniform(0.05, 1.2, n)
    D = np.exp(-0.04 * T)
    is_call = K >= F
    p = black_price(F, K, T, sig, D, is_call)
    iv = black_implied_vol(p, F, K, T, D, is_call)
    ok = p > 1e-10
    np.testing.assert_allclose(iv[ok], sig[ok], atol=1e-8)


def test_black_iv_nan_outside_arbitrage_bounds():
    D = np.exp(-0.04)
    # Below intrinsic, and above the forward-value upper bound.
    iv = black_implied_vol(np.array([5.0, 120.0]), 110.0, 100.0, 1.0, D,
                           np.array([True, True]))
    assert np.isnan(iv).all()


# ── Forwards ─────────────────────────────────────────────────────────────


def test_parity_forward_recovers_dividend_adjusted_forward():
    chain = make_chain(r=0.04, q=0.03)
    _, forwards = S.otm_quotes(chain, 100.0, lambda T: 0.04)
    for T, (F, _D, source) in forwards.items():
        assert source == "parity"
        assert F == pytest.approx(100.0 * np.exp((0.04 - 0.03) * T), rel=1e-6)


def test_naive_carry_forward_would_be_wrong():
    """Guard the reason parity exists: S·e^{rT} misses the dividend."""
    T = 2.0
    assert abs(np.exp(0.04 * T) / np.exp(0.01 * T) - 1) > 0.05


def test_carry_fallback_without_pairs():
    chain = make_chain()
    calls_only = chain[chain["option_type"] == "call"]
    _, forwards = S.otm_quotes(calls_only, 100.0, lambda T: 0.04)
    assert all(src == "carry" for _, _, src in forwards.values())


def test_otm_quotes_keep_only_otm_side():
    q, forwards = S.otm_quotes(make_chain(), 100.0, lambda T: 0.04)
    assert (q.loc[q["is_call"], "k"] >= -1e-12).all()
    assert (q.loc[~q["is_call"], "k"] < 1e-12).all()


def test_otm_quotes_drop_one_sided_markets():
    chain = make_chain()
    chain.loc[chain.index[:50], "bid"] = 0.0
    q, _ = S.otm_quotes(chain, 100.0, lambda T: 0.04)
    q_full, _ = S.otm_quotes(make_chain(), 100.0, lambda T: 0.04)
    assert len(q) < len(q_full)


def test_quoted_iv_fallback_when_prices_unusable():
    chain = make_chain()
    chain["bid"] = 0.0
    chain["ask"] = 0.0
    q, forwards = S.otm_quotes(chain, 100.0, lambda T: 0.04)
    assert len(q) > 0
    assert all(src == "carry" for _, _, src in forwards.values())


# ── SVI fitting ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
def test_box_lsq3_is_exact(seed):
    """The active-set solver must match a general bounded solver."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(40, 3))
    b = rng.normal(size=40) * 3
    x, f = S._box_lsq3(A.T @ A, A.T @ b, float(b @ b))
    ref = lsq_linear(A, b, bounds=(S._LOWER, S._UPPER)).x
    f_ref = float(np.sum((A @ ref - b) ** 2))
    assert f == pytest.approx(f_ref, rel=1e-8, abs=1e-10)
    assert np.all(x >= S._LOWER - 1e-12) and np.all(x <= S._UPPER + 1e-12)


def test_svi_recovers_exact_svi_slice():
    true = dict(a=0.01, b=0.1, rho=-0.6, m=0.02, sigma=0.15)
    k = np.linspace(-0.5, 0.3, 40)
    w = S.svi_total_variance(k, **true)
    p = S.fit_svi_quasi_explicit(k, w)
    np.testing.assert_allclose(S.svi_total_variance(k, **p), w, atol=1e-7)


def test_svi_fit_respects_constraints():
    rng = np.random.default_rng(3)
    k = np.linspace(-0.3, 0.2, 25)
    w = np.abs(0.01 + 0.05 * k ** 2 + rng.normal(0, 0.003, k.size))
    p = S.fit_svi_quasi_explicit(k, w)
    assert p["a"] >= 0 and p["b"] >= 0 and abs(p["rho"]) <= 1
    assert p["b"] * (1 + abs(p["rho"])) <= S.MAX_WING_SLOPE + 1e-9


def test_fit_slices_recovers_smile():
    chain = make_chain(q=0.02)
    fits = S.fit_slices(chain, 100.0, 0.04)
    assert len(fits) == 5
    for f in fits:
        k = np.linspace(f.k_min, f.k_max, 15)
        np.testing.assert_allclose(f.iv(k), smile_iv(k, f.T), atol=0.004)


def test_outlier_quote_is_rejected():
    chain = make_chain(tenors=(0.25,), n_strikes=30)
    # Corrupt one OTM put far from the bid-ask: price it at 1.5x.
    puts = chain.index[(chain["option_type"] == "put") & (chain["strike"] < 90)]
    i = puts[len(puts) // 2]
    for col in ("bid", "ask", "mid"):
        chain.loc[i, col] *= 1.5
    (fit,) = S.fit_slices(chain, 100.0, 0.04)
    assert fit.n_outliers >= 1
    assert fit.rmse_iv < 0.004


# ── Slices → grid ────────────────────────────────────────────────────────


def _flat_slice(T, vol, k_min=-0.2, k_max=0.1):
    return S.SliceFit(T=T, forward=100.0, discount=1.0, forward_source="parity",
                      n_quotes=10, n_outliers=0, k_min=k_min, k_max=k_max,
                      a=vol ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1, rmse_iv=0.0)


def test_expiries_are_interpolated_not_pooled():
    """Total variance must be linear in T between the bracketing expiries."""
    s1, s2 = _flat_slice(0.05, 0.30), _flat_slice(0.15, 0.20)
    iv, obs = S.slices_to_grid([s1, s2], np.array([0.0]), np.array([0.10]))
    w = 0.5 * (0.30 ** 2 * 0.05) + 0.5 * (0.20 ** 2 * 0.15)
    assert iv[0, 0] == pytest.approx(np.sqrt(w / 0.10))
    assert obs[0, 0]


def test_flat_vol_beyond_last_expiry_is_flagged():
    s1, s2 = _flat_slice(0.1, 0.2), _flat_slice(0.5, 0.25)
    iv, obs = S.slices_to_grid([s1, s2], np.array([0.0]), np.array([2.0]))
    assert iv[0, 0] == pytest.approx(0.25)
    assert not obs[0, 0]


def test_observed_mask_follows_quoted_range():
    s1 = _flat_slice(0.1, 0.2, k_min=-0.1, k_max=0.05)
    s2 = _flat_slice(0.3, 0.2, k_min=-0.2, k_max=0.10)
    k = np.array([-0.15, -0.05, 0.0, 0.08])
    _, obs = S.slices_to_grid([s1, s2], k, np.array([0.2]))
    # Inside both ranges only where k in [-0.1, 0.05].
    assert obs[:, 0].tolist() == [False, True, True, False]


def test_sparse_slice_extrapolation_stays_bounded():
    """A steep SVI wing must not explode beyond the quoted strikes."""
    steep = S.SliceFit(T=0.1, forward=50.0, discount=1.0, forward_source="carry",
                       n_quotes=7, n_outliers=0, k_min=-0.09, k_max=0.05,
                       a=0.0, b=1.0, rho=-0.99, m=-0.137, sigma=0.0226, rmse_iv=0.002)
    pure = np.sqrt(steep.total_variance(np.array([-0.4])) / 0.1)[0]
    capped = np.sqrt(steep.total_variance_extrapolated(np.array([-0.4])) / 0.1)[0]
    assert pure > 2.0          # the raw SVI wing really is wild here
    assert capped < 0.75 * pure


def test_extrapolation_never_lowers_variance_in_the_wings():
    f = S.fit_slices(make_chain(tenors=(0.25,)), 100.0, 0.04)[0]
    k_out = np.linspace(f.k_max, f.k_max + 1.0, 20)
    w = f.total_variance_extrapolated(k_out)
    assert np.all(np.diff(w) >= -1e-12)
    k_out = np.linspace(f.k_min - 1.0, f.k_min, 20)
    w = f.total_variance_extrapolated(k_out)
    assert np.all(np.diff(w) <= 1e-12)


# ── VolSurface integration ───────────────────────────────────────────────


def test_from_chain_records_builder_mask_and_slices():
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(make_chain(), ticker="T", as_of=date(2026, 1, 6),
                               spot=100.0, r=0.04)
    assert vs.builder == S.BUILDER_VERSION
    assert vs.observed is not None and vs.observed.shape == vs.iv_grid.shape
    assert len(vs.slices) == 5
    assert vs.atm_vol(0.25) == pytest.approx(smile_iv(0.0, 0.25), abs=0.005)


def test_save_load_preserves_mask_and_slices(tmp_path):
    from src.surface.surface import VolSurface

    vs = VolSurface.from_chain(make_chain(), ticker="T", as_of=date(2026, 1, 6),
                               spot=100.0, r=0.04)
    vs2 = VolSurface.load(vs.save(tmp_path / "s.parquet"))
    np.testing.assert_array_equal(vs2.observed, vs.observed)
    assert vs2.builder == vs.builder
    assert vs2.slices == vs.slices


def test_delta_strike_on_flat_surface():
    from scipy.stats import norm

    from src.surface.surface import VolSurface

    sig, T = 0.2, 0.25
    vs = VolSurface(ticker="F", as_of=date(2026, 1, 6),
                    k_grid=np.linspace(-0.5, 0.5, 21), t_grid=np.array([0.1, 0.25, 1.0]),
                    iv_grid=np.full((21, 3), sig))
    z = norm.ppf(0.75)
    assert vs.delta_strike(-0.25, T) == pytest.approx(-z * sig * np.sqrt(T) + 0.5 * sig ** 2 * T)
    assert vs.delta_strike(0.25, T) == pytest.approx(z * sig * np.sqrt(T) + 0.5 * sig ** 2 * T)
    assert vs.skew(T) == pytest.approx(0.0, abs=1e-12)


# ── Diagnostics ──────────────────────────────────────────────────────────


def test_variance_swap_vol_of_flat_smile_is_the_vol():
    from src.surface.diagnostics import variance_swap_vol

    slices = [_flat_slice(T, 0.2, k_min=-3, k_max=2).to_dict() for T in (0.05, 0.15)]
    assert variance_swap_vol(slices, 0.1) == pytest.approx(0.2, abs=1e-4)


def test_variance_swap_exceeds_atm_under_skew():
    from src.surface.diagnostics import variance_swap_vol

    fits = S.fit_slices(make_chain(), 100.0, 0.04)
    vs_vol = variance_swap_vol([f.to_dict() for f in fits], 0.25)
    assert vs_vol > smile_iv(0.0, 0.25)


def test_noise_stats_detects_mean_reverting_noise():
    from src.surface.diagnostics import noise_stats

    rng = np.random.default_rng(0)
    clean = pd.Series(np.cumsum(rng.normal(0, 0.5, 300)))
    noisy = clean + rng.normal(0, 2.0, 300)
    assert noise_stats(noisy)["lag1_autocorr"] < -0.3
    assert abs(noise_stats(clean)["lag1_autocorr"]) < 0.2
