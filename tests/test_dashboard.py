"""Tests for the dashboard: data shaping, theme, and a full-app smoke run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard import data as D
from tests.synthetic_chains import smile_iv


@pytest.fixture
def dash_store(stores):
    """The shared synthetic store plus a built feature table."""
    from src.features.table import build_feature_table, save_feature_table

    feats = build_feature_table(surfaces_dir=stores["surfaces_dir"],
                                underlying_dir=stores["underlying_dir"],
                                vix_dir=stores["vix_dir"])
    save_feature_table(feats, f"{stores['root']}/features/surface_features.parquet")
    return {**stores, "features": feats}


@pytest.fixture
def surface(dash_store):
    return D.load_vol_surface("TEST", dash_store["dates"][-1], dash_store["surfaces_dir"])


# ── Feature-table views ──────────────────────────────────────────────────


def test_on_or_before():
    dates = [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-04")]
    assert D.on_or_before(dates, pd.Timestamp("2026-03-03")) == dates[0]
    assert D.on_or_before(dates, pd.Timestamp("2026-03-01")) is None


def test_overview_is_one_row_per_ticker_at_as_of(dash_store):
    f = dash_store["features"]
    as_of = pd.Timestamp(dash_store["dates"][2])
    ov = D.overview(f, as_of)
    assert ov["ticker"].tolist() == ["TEST"]
    assert ov["date"].iloc[0] == as_of
    assert len(ov["atm_30d_trend"].iloc[0]) == 3       # only rows up to as_of


def test_overview_drops_stale_tickers(dash_store):
    f = dash_store["features"]
    later = pd.Timestamp(dash_store["dates"][-1]) + pd.Timedelta(days=30)
    assert D.overview(f, later).empty


def test_history_window(dash_store):
    f = dash_store["features"]
    d = [pd.Timestamp(x) for x in dash_store["dates"]]
    h = D.history(f, "TEST", start=d[1], end=d[3])
    assert h["date"].tolist() == d[1:4]


# ── Surface views ────────────────────────────────────────────────────────


def test_load_vol_surface_missing_returns_none(dash_store):
    assert D.load_vol_surface("NOPE", dash_store["dates"][0], dash_store["surfaces_dir"]) is None


def test_default_expiries_pick_nearest_listed(surface):
    tbl = D.expiry_table(surface)
    picks = D.default_expiries(surface, (30, 91))
    for target, e in zip((30, 91), picks):
        days = tbl.set_index("expiration").loc[e, "days"]
        assert days == tbl.iloc[(tbl["days"] - target).abs().argmin()]["days"]


def test_smile_quotes_sit_on_the_fit(dash_store, surface):
    """Quotes are rebuilt on the fit's own forward, so residuals match its RMSE."""
    from src.surface.slices import SliceFit

    e = D.default_expiries(surface)[0]
    q = D.smile_quotes(surface, "TEST", dash_store["dates"][-1], [e], dash_store["options_dir"])
    s = SliceFit(**next(x for x in surface.slices if x["expiration"] == e))
    resid = s.iv(q["k"].to_numpy()) - q["iv"].to_numpy()
    assert len(q) > 5
    assert np.sqrt(np.mean(resid ** 2)) == pytest.approx(s.rmse_iv, abs=2e-3)
    assert (q["half_spread"] > 0).all()


def test_smile_curves_cover_the_quoted_range(surface):
    e = D.default_expiries(surface)[0]
    c = D.smile_curves(surface, [e])
    s = next(x for x in surface.slices if x["expiration"] == e)
    assert c["k"].min() < s["k_min"] and c["k"].max() > s["k_max"]
    assert np.isfinite(c["iv"]).all()


def test_term_structure_passes_through_listed_expiries(surface):
    curve, points = D.term_structure(surface)
    merged = points.merge(curve, on="days", suffixes=("_pt", "_curve"))
    assert len(merged) == len(points)
    np.testing.assert_allclose(merged["iv_pt"], merged["iv_curve"])


def test_delta_tenor_grid(surface):
    g = D.delta_tenor_grid(surface)
    assert list(g.columns) == list(D.DELTA_COLUMNS)
    assert g.loc["30d", "ATM"] == pytest.approx(smile_iv(0.0, 30 / 365, atm=0.24), abs=0.006)
    # Shortest synthetic expiry is ~22 days: short tenors are blank, not invented.
    assert g.loc[["7d", "14d"]].isna().all().all()
    # Put skew: the 25Δ put is richer than the 25Δ call.
    assert g.loc["91d", "25Δ put"] > g.loc["91d", "25Δ call"]


def test_formatters():
    assert D.pct(0.1234) == "12.3%"
    assert D.pts(-0.028) == "-2.8 pts"
    assert D.pct(float("nan")) == "–"


# ── Theme ────────────────────────────────────────────────────────────────


def test_theme_selection_and_scales():
    from src.dashboard.theme import DARK, LIGHT, colorscale, theme_for

    assert theme_for("dark") is DARK and theme_for("light") is LIGHT and theme_for(None) is LIGHT
    cs = colorscale(LIGHT.diverging)
    assert cs[0][0] == 0 and cs[-1][0] == 1 and cs[1] == [0.5, LIGHT.diverging[1]]
    # The sequential ramp's low end recedes into the surface in both modes.
    assert LIGHT.sequential[0] == DARK.sequential[-1]


# ── Whole app ────────────────────────────────────────────────────────────


def test_app_runs_without_errors(dash_store, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VSS_DATA_DIR", dash_store["root"])
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "TEST 30d ATM" in labels and "VIX" in labels
    assert at.multiselect[0].value          # smile expiries defaulted


def test_app_explains_an_empty_store(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VSS_DATA_DIR", str(tmp_path))
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("build_features" in i.value for i in at.info)


# ── Forecast alignment ───────────────────────────────────────────────────


def test_window_end_counts_trading_days_across_holidays():
    # Fri 2026-11-20 + 5 sessions skips Thanksgiving (Thu 11-26): Mon..Wed, Fri, Mon.
    assert D.window_end([pd.Timestamp("2026-11-20")], 5).iloc[0] == pd.Timestamp("2026-11-30")
    assert D.window_end([pd.Timestamp("2026-10-02")], 21).iloc[0] == pd.Timestamp("2026-11-02")


def test_forecast_outcomes_only_completed_windows():
    hist = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-10-01"]),
        "forecast_vol": [0.15, 0.16, 0.14], "lo80_vol": 0.1, "hi80_vol": 0.2,
        "implied_vol": 0.17, "realised_vol": [0.12, 0.13, np.nan],
    })
    out = D.forecast_outcomes(hist, 21)
    assert len(out) == 2
    assert (out["window_end"] > out["date"]).all()
