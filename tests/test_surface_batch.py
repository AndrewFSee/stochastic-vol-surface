"""Tests for the batch surface-construction layer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.surface import batch as B


# ── Synthetic chain fixture ──────────────────────────────────────────────


def _synthetic_chain(ticker="TEST", as_of=date(2026, 1, 6), spot=100.0):
    """Build a small but well-formed options chain with a realistic smile."""
    rows = []
    for T in (0.0833, 0.25, 0.5, 1.0, 2.0):
        expiry = pd.Timestamp(as_of) + pd.Timedelta(days=int(T * 365))
        for k in np.linspace(-0.35, 0.18, 22):
            strike = spot * np.exp(k)
            # Convex smile in log-moneyness, mild term structure.
            iv = 0.20 + 0.35 * k**2 - 0.05 * k + 0.01 * T
            for opt in ("call", "put"):
                rows.append({
                    "ticker": ticker,
                    "as_of": pd.Timestamp(as_of),
                    "expiration": expiry,
                    "strike": float(strike),
                    "option_type": opt,
                    "bid": 1.0, "ask": 1.1, "mid": 1.05, "last_price": 1.05,
                    "volume": 100.0, "open_interest": 500.0,
                    "implied_volatility_market": float(iv),
                    "T": float(T),
                    "underlying_price": spot,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    """A raw options store containing three snapshots for one ticker."""
    opt_dir = tmp_path / "options"
    dates = [date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    for d in dates:
        part = opt_dir / "ticker=TEST" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        _synthetic_chain(as_of=d).to_parquet(part / "chain.parquet", index=False)
    return {
        "options_dir": str(opt_dir),
        "surfaces_dir": str(tmp_path / "surfaces"),
        "dates": dates,
    }


# ── Discovery ────────────────────────────────────────────────────────────


def test_available_tickers(store):
    assert B.available_tickers(store["options_dir"]) == ["TEST"]


def test_available_dates(store):
    assert B.available_dates("TEST", store["options_dir"]) == store["dates"]


def test_available_dates_unknown_ticker(store):
    assert B.available_dates("NOPE", store["options_dir"]) == []


def test_surface_path_layout():
    """Must match the layout scripts/train.py globs for."""
    p = B.surface_path("SPY", date(2026, 1, 6), "data/surfaces")
    assert p.as_posix().endswith("data/surfaces/ticker=SPY/date=2026-01-06/surface.parquet")


# ── Grid scoring ─────────────────────────────────────────────────────────


def test_score_accepts_plausible_grid():
    ok, msg, stats = B.score_grid(np.full((25, 8), 0.20))
    assert ok and msg == ""
    assert stats["iv_min"] == pytest.approx(0.20)


def test_score_rejects_all_nan():
    ok, msg, _ = B.score_grid(np.full((25, 8), np.nan))
    assert not ok and "non-finite" in msg


def test_score_rejects_mostly_nan():
    g = np.full((25, 8), 0.2)
    g[:20, :] = np.nan
    ok, msg, _ = B.score_grid(g)
    assert not ok and "NaN" in msg


def test_score_rejects_implausibly_high_iv():
    g = np.full((25, 8), 0.2)
    g[0, 0] = 12.0
    ok, msg, _ = B.score_grid(g)
    assert not ok and "max IV" in msg


def test_score_rejects_nonpositive_iv():
    g = np.full((25, 8), 0.2)
    g[0, 0] = 0.0
    ok, msg, _ = B.score_grid(g)
    assert not ok and "min IV" in msg


def test_score_reports_nan_fraction():
    g = np.full((25, 8), 0.2)
    g[0, :] = np.nan
    _, _, stats = B.score_grid(g)
    assert stats["nan_fraction"] == pytest.approx(8 / 200)


# ── Single build ─────────────────────────────────────────────────────────


def test_build_one_writes_surface(store):
    res = B.build_one("TEST", store["dates"][0], **_dirs(store))
    assert res.status == "built", res.message
    assert res.path.exists()


def test_build_one_result_is_populated(store):
    res = B.build_one("TEST", store["dates"][0], **_dirs(store))
    assert res.n_chain_rows > 0
    assert res.spot == pytest.approx(100.0)
    assert 0.0 < res.atm_3m < 3.0
    assert np.isfinite(res.iv_min) and np.isfinite(res.iv_max)


def test_build_one_skips_existing(store):
    B.build_one("TEST", store["dates"][0], **_dirs(store))
    again = B.build_one("TEST", store["dates"][0], **_dirs(store))
    assert again.status == "skipped"


def test_build_one_overwrite_rebuilds(store):
    B.build_one("TEST", store["dates"][0], **_dirs(store))
    again = B.build_one("TEST", store["dates"][0], overwrite=True, **_dirs(store))
    assert again.status == "built"


def test_build_one_missing_date_fails_gracefully(store):
    res = B.build_one("TEST", date(2020, 1, 1), **_dirs(store))
    assert res.status == "failed"
    assert res.path is None


def test_build_one_uses_supplied_rate_history(store):
    """The discount rate must come from the curve, not a hard-coded constant."""
    hist = pd.DataFrame(
        {"3M": [0.0777], "1Y": [0.0777]},
        index=pd.to_datetime(["2026-01-01"]),
    ).rename_axis("date")
    res = B.build_one("TEST", store["dates"][0], rates_history=hist, **_dirs(store))
    assert res.r == pytest.approx(0.0777)


def test_build_one_falls_back_without_history(store):
    res = B.build_one("TEST", store["dates"][0],
                      rates_history=pd.DataFrame(), **_dirs(store))
    assert np.isfinite(res.r)


def _dirs(store):
    return {
        "options_dir": store["options_dir"],
        "surfaces_dir": store["surfaces_dir"],
    }


# ── Corpus sweep ─────────────────────────────────────────────────────────


def test_build_corpus_builds_every_date(store):
    rep = B.build_corpus(
        options_dir=store["options_dir"], surfaces_dir=store["surfaces_dir"],
        progress=False,
    )
    assert rep.counts().get("built") == 3


def test_build_corpus_is_resumable(store):
    kw = dict(options_dir=store["options_dir"],
              surfaces_dir=store["surfaces_dir"], progress=False)
    B.build_corpus(**kw)
    second = B.build_corpus(**kw)
    assert second.counts().get("skipped") == 3
    assert "built" not in second.counts()


def test_build_corpus_respects_date_range(store):
    rep = B.build_corpus(
        options_dir=store["options_dir"], surfaces_dir=store["surfaces_dir"],
        start="2026-01-07", progress=False,
    )
    assert rep.counts().get("built") == 2


def test_build_corpus_report_frame(store):
    rep = B.build_corpus(
        options_dir=store["options_dir"], surfaces_dir=store["surfaces_dir"],
        progress=False,
    )
    df = rep.to_frame()
    assert len(df) == 3
    assert {"ticker", "as_of", "status", "atm_3m"} <= set(df.columns)


def test_build_corpus_summary_is_readable(store):
    rep = B.build_corpus(
        options_dir=store["options_dir"], surfaces_dir=store["surfaces_dir"],
        progress=False,
    )
    assert "Surface corpus" in rep.summary()


def test_build_corpus_empty_store(tmp_path):
    rep = B.build_corpus(options_dir=str(tmp_path / "none"), progress=False)
    assert rep.results == []


# ── Reading back ─────────────────────────────────────────────────────────


def test_load_surface_history_shapes(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    dates, k, t, grids = B.load_surface_history("TEST", store["surfaces_dir"])

    assert len(dates) == 3
    assert grids.shape == (3, len(k), len(t))
    assert grids.shape[1:] == (25, 8)          # the configured standard grid


def test_load_surface_history_is_ordered(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    dates, _, _, _ = B.load_surface_history("TEST", store["surfaces_dir"])
    assert dates == sorted(dates)


def test_load_surface_history_values_are_finite(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    _, _, _, grids = B.load_surface_history("TEST", store["surfaces_dir"])
    assert np.isfinite(grids).all()


def test_load_surface_history_date_filter(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    dates, _, _, grids = B.load_surface_history(
        "TEST", store["surfaces_dir"], start="2026-01-07",
    )
    assert len(dates) == 2 and grids.shape[0] == 2


def test_load_surface_history_missing_ticker(store):
    dates, k, t, grids = B.load_surface_history("NOPE", store["surfaces_dir"])
    assert dates == [] and grids.size == 0


def test_load_surfaces_returns_dict_keyed_by_date(store):
    """The backtest engine consumes {date: VolSurface}."""
    from src.surface.surface import VolSurface

    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    surfaces = B.load_surfaces("TEST", store["surfaces_dir"])

    assert set(surfaces) == set(store["dates"])
    assert all(isinstance(v, VolSurface) for v in surfaces.values())


def test_load_surfaces_are_queryable(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    vs = B.load_surfaces("TEST", store["surfaces_dir"])[store["dates"][0]]
    assert 0.0 < vs.atm_vol(0.25) < 3.0


def test_load_surfaces_date_filter(store):
    B.build_corpus(options_dir=store["options_dir"],
                   surfaces_dir=store["surfaces_dir"], progress=False)
    surfaces = B.load_surfaces("TEST", store["surfaces_dir"], start="2026-01-07")
    assert len(surfaces) == 2


def test_load_surfaces_missing_ticker(store):
    assert B.load_surfaces("NOPE", store["surfaces_dir"]) == {}


def test_roundtrip_grid_matches_built_surface(store):
    """A reloaded grid must equal what the builder produced."""
    from src.surface.surface import VolSurface

    res = B.build_one("TEST", store["dates"][0], **_dirs(store))
    reloaded = VolSurface.load(res.path)
    _, _, _, grids = B.load_surface_history("TEST", store["surfaces_dir"])
    np.testing.assert_allclose(grids[0], reloaded.iv_grid, rtol=1e-9)
