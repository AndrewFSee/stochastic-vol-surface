"""Tests for the feature layer (src.features) and the underlying price store."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features import realized as R
from src.features.surface_features import TENORS_DAYS, surface_features
from src.surface.slices import ExpirySurface, fit_slices
from tests.synthetic_chains import make_chain, smile_iv


# ── Realised vol ─────────────────────────────────────────────────────────


def _gbm_prices(sigma=0.2, n=600, seed=0, start="2025-01-02"):
    """Daily OHLC from a GBM sampled intraday, so OHLC estimators have signal."""
    rng = np.random.default_rng(seed)
    steps = 390  # one per minute; coarser sampling understates high-low ranges
    dt = 1 / (252 * steps)
    log_p = np.cumsum(rng.normal(-0.5 * sigma ** 2 * dt, sigma * np.sqrt(dt), n * steps))
    path = 100 * np.exp(log_p).reshape(n, steps)
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({
        "open": path[:, 0], "high": path.max(axis=1), "low": path.min(axis=1),
        "close": path[:, -1], "adj_close": path[:, -1], "volume": 1e6,
    }, index=idx)


def test_close_to_close_vol_is_zero_mean_rms():
    p = pd.Series([100.0, 101.0, 99.0, 100.0])
    r = np.log(p).diff().dropna()
    expected = np.sqrt(252 * np.mean(r ** 2))
    assert R.close_to_close_vol(p, 3).iloc[-1] == pytest.approx(expected)


def test_realised_vol_recovers_gbm_sigma():
    f = R.realized_features(_gbm_prices(sigma=0.25))
    assert f["rv_cc_63d"].dropna().mean() == pytest.approx(0.25, abs=0.02)
    assert f["rv_yz_21d"].dropna().mean() == pytest.approx(0.25, abs=0.03)


def test_yang_zhang_is_less_noisy_than_close_to_close():
    f = R.realized_features(_gbm_prices(sigma=0.25, n=1500, seed=1))
    assert f["rv_yz_21d"].std() < f["rv_cc_21d"].std()


def test_missing_adj_close_falls_back_to_close():
    p = _gbm_prices(n=30)
    p.iloc[-1, p.columns.get_loc("adj_close")] = np.nan
    f = R.realized_features(p)
    assert np.isfinite(f["ret_1d"].iloc[-1])


# ── Surface features ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def slices():
    chain = make_chain(r=0.04, q=0.02, tenors=(0.06, 0.12, 0.25, 0.5, 1.0, 1.5),
                       k_range=(-0.45, 0.25), n_strikes=30)
    return [f.to_dict() for f in fit_slices(chain, 100.0, 0.04)]


def test_atm_features_match_the_smile(slices):
    f = surface_features(slices)
    for name in ("30d", "91d", "365d"):
        T = TENORS_DAYS[name] / 365
        # Linear-in-total-variance interpolation between expiries is not the
        # generator's functional form, so allow a fraction of a vol point.
        assert f[f"atm_{name}"] == pytest.approx(smile_iv(0.0, T), abs=0.004)


def test_put_skew_gives_negative_risk_reversal(slices):
    f = surface_features(slices)
    assert f["rr25_30d"] < 0 and f["rr10_30d"] < f["rr25_30d"]
    assert f["bf25_30d"] > 0
    assert f["atm_skew_30d"] < 0


def test_unbracketed_tenor_is_nan_not_extrapolated(slices):
    """The shortest expiry is ~22 days, so 7d and 14d must be NaN."""
    f = surface_features(slices)
    assert np.isnan(f["atm_7d"]) and np.isnan(f["ts_7_30"])
    g = surface_features(slices, allow_extrapolation=True)
    assert np.isfinite(g["atm_7d"])


def test_delta_strike_has_the_requested_delta(slices):
    from scipy.special import ndtr

    surf = ExpirySurface(slices)
    T = 30 / 365
    for delta in (-0.25, -0.10, 0.25, 0.10):
        k = surf.delta_strike(delta, T)
        sw = np.sqrt(surf.total_variance(k, T))
        call_delta = ndtr(-k / sw + 0.5 * sw)
        assert call_delta == pytest.approx(delta if delta > 0 else 1 + delta, abs=1e-6)


def test_variance_swap_exceeds_atm_with_skew(slices):
    f = surface_features(slices)
    assert f["vs_30d"] > f["atm_30d"]


def test_implied_carry_is_rate_minus_dividend(slices):
    f = surface_features(slices, spot=100.0)
    assert f["fwd_carry_1y"] == pytest.approx(0.04 - 0.02, abs=1e-4)


def test_quality_columns(slices):
    f = surface_features(slices)
    assert f["n_expiries"] == len(slices)
    assert f["parity_fraction"] == 1.0
    assert f["fit_rmse"] < 0.005


def test_too_few_slices_gives_no_features(slices):
    assert surface_features(slices[:1]) == {}


# ── Dynamics & point-in-time safety ──────────────────────────────────────


def _panel(n=120, tickers=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n)
    rows = []
    for t in tickers:
        x = 0.2 + np.cumsum(rng.normal(0, 0.005, n))
        rows += [{"ticker": t, "date": d, "atm_30d": v} for d, v in zip(dates, x)]
    return pd.DataFrame(rows)


def test_dynamics_are_point_in_time():
    """Appending later rows must not change any earlier row's features."""
    from src.features.table import add_dynamics

    full = add_dynamics(_panel(), columns=["atm_30d"])
    cutoff = full["date"].sort_values().unique()[80]
    trunc = add_dynamics(_panel().query("date <= @cutoff"), columns=["atm_30d"])
    cols = ["atm_30d_d1", "atm_30d_d5", "atm_30d_z63", "atm_30d_pct252"]
    a = full[full["date"] <= cutoff].sort_values(["ticker", "date"])[cols].to_numpy()
    b = trunc.sort_values(["ticker", "date"])[cols].to_numpy()
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_missing_day_breaks_the_change_instead_of_spanning_it():
    from src.features.table import add_dynamics

    df = _panel(n=30, tickers=("AAA",))
    gap_day = df["date"].iloc[3]          # 2026-01-08, a regular session
    out = add_dynamics(df[df["date"] != gap_day], columns=["atm_30d"])
    after_gap = out[out["date"] == df["date"].iloc[4]]
    assert np.isnan(after_gap["atm_30d_d1"].iloc[0])


def test_dynamics_are_per_ticker():
    from src.features.table import add_dynamics

    out = add_dynamics(_panel(n=30), columns=["atm_30d"])
    first = out.sort_values("date").groupby("ticker").head(1)
    assert first["atm_30d_d1"].isna().all()


# ── Snapshot closes ──────────────────────────────────────────────────────


def test_fill_closes_from_snapshots():
    from src.features.table import fill_closes_from_snapshots

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-05"), "AAA"), (pd.Timestamp("2026-01-06"), "AAA")],
        names=["date", "ticker"])
    prices = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                           "close": [10.0, np.nan], "adj_close": [10.0, np.nan],
                           "volume": 1.0}, index=idx)
    spots = pd.Series({(pd.Timestamp("2026-01-06"), "AAA"): 11.0,
                       (pd.Timestamp("2026-01-07"), "AAA"): 12.0})
    spots.index.names = ["date", "ticker"]
    out = fill_closes_from_snapshots(prices, spots)
    assert out.loc[(pd.Timestamp("2026-01-05"), "AAA"), "close"] == 10.0   # untouched
    assert out.loc[(pd.Timestamp("2026-01-06"), "AAA"), "close"] == 11.0   # filled
    assert out.loc[(pd.Timestamp("2026-01-07"), "AAA"), "close"] == 12.0   # added


# ── Underlying store ─────────────────────────────────────────────────────


def _price_frame(dates, close, ticker="AAA"):
    idx = pd.MultiIndex.from_product([pd.to_datetime(dates), [ticker]],
                                     names=["date", "ticker"])
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "adj_close": close, "volume": 1.0}, index=idx)


def test_underlying_store_merges_and_prefers_new_values(tmp_path):
    from src.data.underlying import load_underlying_history, save_underlying_history

    d = str(tmp_path)
    save_underlying_history(_price_frame(["2026-01-05", "2026-01-06"], [1.0, np.nan]), d)
    save_underlying_history(_price_frame(["2026-01-06", "2026-01-07"], [2.0, 3.0]), d)
    out = load_underlying_history(d)
    assert out["close"].tolist() == [1.0, 2.0, 3.0]


def test_underlying_store_is_idempotent(tmp_path):
    from src.data.underlying import load_underlying_history, save_underlying_history

    d = str(tmp_path)
    df = _price_frame(["2026-01-05", "2026-01-06"], [1.0, 2.0])
    save_underlying_history(df, d)
    save_underlying_history(df, d)
    assert len(load_underlying_history(d)) == 2


# ── End to end ───────────────────────────────────────────────────────────


@pytest.fixture
def stores(tmp_path):
    """Options, surfaces, prices and VIX for one ticker over five days."""
    from src.data.underlying import save_underlying_history
    from src.surface.batch import build_corpus

    dates = list(pd.bdate_range("2026-03-02", periods=5).date)
    opt = tmp_path / "options"
    for i, d in enumerate(dates):
        part = opt / "ticker=TEST" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        make_chain(as_of=d, atm=0.20 + 0.01 * i,
                   tenors=(0.06, 0.12, 0.25, 0.5, 1.0)).to_parquet(
            part / "chain.parquet", index=False)
    build_corpus(options_dir=str(opt), surfaces_dir=str(tmp_path / "surfaces"),
                 progress=False)

    prices = _gbm_prices(n=80, start="2025-11-03").loc[:pd.Timestamp(dates[-1])]
    prices["ticker"] = "TEST"
    save_underlying_history(prices.reset_index().set_index(["date", "ticker"]),
                            str(tmp_path / "underlying"))

    vix_dir = tmp_path / "vix"
    vix_dir.mkdir()
    pd.DataFrame({"VIX": [15.0 + i for i in range(5)]},
                 index=pd.DatetimeIndex(pd.to_datetime(dates), name="date")).to_parquet(
        vix_dir / f"vix_{dates[-1].isoformat()}.parquet")
    return {"surfaces_dir": str(tmp_path / "surfaces"),
            "underlying_dir": str(tmp_path / "underlying"),
            "vix_dir": str(vix_dir), "dates": dates}


def test_build_feature_table_end_to_end(stores, tmp_path):
    from src.features.table import (FEATURE_VERSION, build_feature_table,
                                    load_feature_table, save_feature_table)

    df = build_feature_table(surfaces_dir=stores["surfaces_dir"],
                             underlying_dir=stores["underlying_dir"],
                             vix_dir=stores["vix_dir"])
    assert len(df) == 5 and not df.duplicated(["ticker", "date"]).any()
    assert df["atm_30d"].is_monotonic_increasing          # atm rose by 1 pt a day
    assert df["rv_cc_21d"].notna().all() and df["vrp_30d"].notna().all()
    assert df["mkt_vix"].tolist() == [15.0, 16.0, 17.0, 18.0, 19.0]
    assert (df["feature_version"] == FEATURE_VERSION).all()
    assert df["atm_30d_d1"].iloc[1:].round(4).eq(0.01).all()

    path = save_feature_table(df, str(tmp_path / "f.parquet"))
    back = load_feature_table(str(path), start=str(stores["dates"][2]))
    assert len(back) == 3


def test_exchange_holiday_is_not_a_gap():
    """2026-01-19 (MLK Day) is not a session, so no row there is expected."""
    from src.features.table import add_dynamics

    df = _panel(n=30, tickers=("AAA",))
    holiday = pd.Timestamp("2026-01-19")
    out = add_dynamics(df[df["date"] != holiday], columns=["atm_30d"])
    after = out[out["date"] == pd.Timestamp("2026-01-20")]
    assert np.isfinite(after["atm_30d_d1"].iloc[0])
