"""Earnings dates, the implied earnings move and the event-adjusted forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.events import event_sessions, save_earnings, load_earnings
from src.features.earnings import (calendar_columns, historical_event_variance,
                                   implied_event_variance)

SESSIONS = pd.bdate_range("2026-01-02", "2026-12-31")


def _earn(*stamps, ticker="AAA"):
    return pd.DataFrame({"ticker": ticker, "announced": pd.to_datetime(list(stamps)).tz_localize(
        "America/New_York")})


def test_release_timing_picks_the_session_that_moves():
    ev = event_sessions(_earn("2026-04-30 16:00",      # Thu after close -> Fri
                              "2026-07-28 07:00",      # Tue before open -> Tue
                              "2026-10-30 16:30"),     # Fri after close -> Mon
                        SESSIONS)["AAA"]
    assert list(ev.strftime("%Y-%m-%d")) == ["2026-05-01", "2026-07-28", "2026-11-02"]


def test_releases_before_the_calendar_are_dropped():
    ev = event_sessions(_earn("2025-10-30 16:00", "2026-04-30 16:00"), SESSIONS)["AAA"]
    assert list(ev.strftime("%Y-%m-%d")) == ["2026-05-01"]


def test_save_replaces_a_tickers_dates_and_keeps_others(tmp_path):
    save_earnings(pd.concat([_earn("2026-04-30 16:00"), _earn("2026-05-05 08:00", ticker="BBB")]),
                  str(tmp_path))
    save_earnings(_earn("2026-05-07 16:00"), str(tmp_path))      # AAA rescheduled
    got = load_earnings(str(tmp_path))
    assert sorted(got["ticker"]) == ["AAA", "BBB"]
    assert got.loc[got["ticker"] == "AAA", "announced"].dt.day.tolist() == [7]


def _term(as_of, expiries, sigma2, e2, event):
    """ATM term with diffusive variance σ² per session plus e² after *event*."""
    base = SESSIONS.searchsorted(pd.Timestamp(as_of), side="right")
    out = []
    for e in expiries:
        n = SESSIONS.searchsorted(pd.Timestamp(e), side="right") - base
        w = sigma2 * n + (e2 if e >= event else 0.0)
        out.append([e, (pd.Timestamp(e) - pd.Timestamp(as_of)).days / 365, w])
    return out


@pytest.mark.parametrize("expiries", [
    ["2026-04-29", "2026-05-01", "2026-05-08"],     # one before, one after
    ["2026-05-01", "2026-05-04"],                   # two after, across a weekend
])
def test_implied_event_variance_recovers_the_jump(expiries):
    s2, e2 = 0.02 ** 2, 0.05 ** 2
    term = _term("2026-04-27", expiries, s2, e2, "2026-05-01")
    got = implied_event_variance(term, pd.Timestamp("2026-05-01"),
                                 pd.Timestamp("2026-04-27"), SESSIONS)
    assert got == pytest.approx(e2, rel=1e-9)


def test_implied_event_variance_is_nan_without_a_usable_expiry():
    term = _term("2026-04-27", ["2026-05-01"], 0.02 ** 2, 0.05 ** 2, "2026-05-01")
    assert np.isnan(implied_event_variance(term, pd.Timestamp("2026-05-01"),
                                           pd.Timestamp("2026-04-27"), SESSIONS))
    flat = _term("2026-04-27", ["2026-04-29", "2026-05-08"], 0.02 ** 2, 0.0, "2026-05-01")
    assert np.isnan(implied_event_variance(flat, pd.Timestamp("2026-05-01"),
                                           pd.Timestamp("2026-04-27"), SESSIONS))


def test_calendar_columns_count_sessions():
    events = pd.DatetimeIndex(["2026-05-01", "2026-07-28"])
    dates = pd.DatetimeIndex(["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-04"])
    c = calendar_columns(dates, events, SESSIONS)
    assert c["earn_days_to"].tolist() == [2, 1, 62, 61]
    assert c["earn_days_since"].tolist()[2:] == [0, 1]
    assert np.isnan(c["earn_days_since"].iloc[0])
    assert c["earn_next_date"].iloc[1] == pd.Timestamp("2026-05-01")


def test_historical_event_variance_measures_the_excess_jump():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=1500)
    r = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    events = days[100::63][:15]
    r[events] = np.where(np.arange(len(events)) % 2, 0.06, -0.06)
    close = 100 * np.exp(r.cumsum())
    hv = historical_event_variance(close, events)
    assert np.isnan(hv[events[2]])                       # fewer than 4 releases seen
    assert hv.iloc[-1] == pytest.approx(0.06 ** 2 - 0.01 ** 2, rel=0.1)


def test_event_adjustment_strips_the_implied_and_adds_the_jump():
    from src.forecast.forecaster import _with_events

    idx = pd.DatetimeIndex(["2026-04-27", "2026-06-01"])
    d = pd.DataFrame({
        "iv_in": [0.40, 0.30], "iv_src": ["vs_30d", "vs_30d"],
        "earn_implied_move": [0.05, np.nan], "ev_hist_var": [0.03 ** 2, 0.03 ** 2],
        "ev_next": pd.to_datetime(["2026-05-01", "2026-07-28"]),
        "ev_in_21": [True, False], "y_21": [0.2, 0.1], "y_21_ex": [0.15, 0.1],
    }, index=idx)
    out = _with_events(d, 21)
    assert out["iv_ex"].iloc[0] == pytest.approx(np.sqrt(0.16 - 0.05 ** 2 * 365 / 30))
    assert out["iv_ex"].iloc[1] == pytest.approx(0.30)          # event beyond the tenor
    assert out["ev_add"].tolist() == pytest.approx([252 / 21 * 0.05 ** 2, 0.0])
    assert out["y_21"].tolist() == [0.15, 0.1] and out["y_true"].tolist() == [0.2, 0.1]


def test_dataset_without_earnings_has_identical_ex_columns(stores):
    from src.forecast.dataset import build_dataset

    ds = build_dataset("TEST", underlying_dir=stores["underlying_dir"],
                       feature_paths=[], events_dir=stores["root"] + "/events")
    assert not ds["ev_day"].any() and not ds["ev_in_21"].any()
    pd.testing.assert_series_equal(ds["y_21_ex"], ds["y_21"], check_names=False)
    assert (ds["rv_m_ex"] - ds["rv_m"]).abs().max() < 1e-12
