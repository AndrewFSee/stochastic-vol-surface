"""Tests for the persistent rate-history store and term interpolation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import rates as R


@pytest.fixture
def curve_history():
    """A two-day, three-tenor curve history."""
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    return pd.DataFrame(
        {"3M": [0.040, 0.041], "1Y": [0.042, 0.043], "10Y": [0.050, 0.051]},
        index=idx,
    ).rename_axis("date")


# ── Persistence ──────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path, curve_history):
    R.save_rates_history(curve_history, rates_dir=str(tmp_path))
    back = R.load_rates_history(rates_dir=str(tmp_path))

    assert list(back.columns) == ["3M", "1Y", "10Y"]
    assert len(back) == 2
    pd.testing.assert_frame_equal(back, curve_history, check_freq=False)


def test_load_missing_history_returns_empty(tmp_path):
    assert R.load_rates_history(rates_dir=str(tmp_path / "nope")).empty


def test_save_merges_without_losing_prior_dates(tmp_path, curve_history):
    R.save_rates_history(curve_history, rates_dir=str(tmp_path))

    newer = pd.DataFrame(
        {"3M": [0.045], "1Y": [0.046], "10Y": [0.055]},
        index=pd.to_datetime(["2026-01-07"]),
    ).rename_axis("date")
    R.save_rates_history(newer, rates_dir=str(tmp_path))

    back = R.load_rates_history(rates_dir=str(tmp_path))
    assert len(back) == 3
    assert back.loc["2026-01-05", "3M"] == pytest.approx(0.040)
    assert back.loc["2026-01-07", "3M"] == pytest.approx(0.045)


def test_save_is_idempotent(tmp_path, curve_history):
    R.save_rates_history(curve_history, rates_dir=str(tmp_path))
    R.save_rates_history(curve_history, rates_dir=str(tmp_path))
    assert len(R.load_rates_history(rates_dir=str(tmp_path))) == 2


def test_save_fills_nulls_from_existing(tmp_path, curve_history):
    """A later write with a NaN cell must not erase a known value."""
    R.save_rates_history(curve_history, rates_dir=str(tmp_path))

    patchy = pd.DataFrame(
        {"3M": [np.nan], "1Y": [0.099], "10Y": [np.nan]},
        index=pd.to_datetime(["2026-01-06"]),
    ).rename_axis("date")
    R.save_rates_history(patchy, rates_dir=str(tmp_path))

    back = R.load_rates_history(rates_dir=str(tmp_path))
    assert back.loc["2026-01-06", "1Y"] == pytest.approx(0.099)   # updated
    assert back.loc["2026-01-06", "3M"] == pytest.approx(0.041)   # preserved


# ── Curve lookup ─────────────────────────────────────────────────────────


def test_get_rate_curve_exact_date(curve_history):
    curve = R.get_rate_curve("2026-01-06", history=curve_history)
    assert curve["3M"] == pytest.approx(0.041)


def test_get_rate_curve_uses_most_recent_prior_date(curve_history):
    """Holidays / publication lag must fall back to the last printed curve."""
    curve = R.get_rate_curve("2026-01-09", history=curve_history)
    assert curve["3M"] == pytest.approx(0.041)


def test_get_rate_curve_before_history_is_empty(curve_history):
    assert R.get_rate_curve("2025-12-01", history=curve_history) == {}


# ── Term interpolation ───────────────────────────────────────────────────


def test_rate_at_exact_tenor(curve_history):
    r = R.get_rate_for_tenor("2026-01-05", 0.25, history=curve_history)
    assert r == pytest.approx(0.040)


def test_rate_interpolates_between_tenors(curve_history):
    """T=0.625y sits midway between the 3M and 1Y knots."""
    r = R.get_rate_for_tenor("2026-01-05", 0.625, history=curve_history)
    assert r == pytest.approx((0.040 + 0.042) / 2, abs=1e-9)
    assert 0.040 < r < 0.042


def test_rate_clamps_below_shortest_tenor(curve_history):
    assert R.get_rate_for_tenor("2026-01-05", 0.001, history=curve_history) == pytest.approx(0.040)


def test_rate_clamps_above_longest_tenor(curve_history):
    assert R.get_rate_for_tenor("2026-01-05", 30.0, history=curve_history) == pytest.approx(0.050)


def test_rate_is_monotone_across_an_upward_curve(curve_history):
    tenors = [0.05, 0.25, 0.5, 1.0, 5.0, 10.0]
    vals = [R.get_rate_for_tenor("2026-01-05", T, history=curve_history) for T in tenors]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_rate_falls_back_when_history_empty():
    r = R.get_rate_for_tenor("2026-01-05", 0.25, history=pd.DataFrame())
    assert r == pytest.approx(R.FALLBACK_RATE)


def test_term_structure_returns_one_rate_per_tenor(curve_history):
    ts = R.get_rate_term_structure("2026-01-05", [0.25, 1.0, 10.0], history=curve_history)
    assert len(ts) == 3
    assert ts.loc[0.25] == pytest.approx(0.040)
    assert ts.loc[10.0] == pytest.approx(0.050)


def test_accepts_date_objects(curve_history):
    r = R.get_rate_for_tenor(date(2026, 1, 6), 0.25, history=curve_history)
    assert r == pytest.approx(0.041)
