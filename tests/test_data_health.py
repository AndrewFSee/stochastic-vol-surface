"""Tests for the data-health reporting layer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import health as H


def _chain(ticker, as_of, n_strikes=30, n_expiries=6, iv=0.2, crossed=False):
    rows = []
    for e in range(n_expiries):
        expiry = pd.Timestamp(as_of) + pd.Timedelta(days=30 * (e + 1))
        for i in range(n_strikes):
            rows.append({
                "ticker": ticker,
                "as_of": pd.Timestamp(as_of),
                "expiration": expiry,
                "strike": 90.0 + i,
                "option_type": "call" if i % 2 else "put",
                "bid": 1.2 if crossed else 1.0,
                "ask": 1.0 if crossed else 1.2,
                "mid": 1.1,
                "last_price": 1.1,
                "volume": 10.0,
                "open_interest": 50.0,
                "implied_volatility_market": iv,
                "T": (e + 1) * 30 / 365,
                "underlying_price": 100.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    """Options store: SPY on 3 consecutive sessions, QQQ missing the middle one."""
    opt = tmp_path / "options"
    sessions = [date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]

    for d in sessions:
        part = opt / "ticker=SPY" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        _chain("SPY", d).to_parquet(part / "chain.parquet", index=False)

    for d in (sessions[0], sessions[2]):
        part = opt / "ticker=QQQ" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        _chain("QQQ", d).to_parquet(part / "chain.parquet", index=False)

    return {"options_dir": str(opt), "sessions": sessions, "root": tmp_path}


# ── Coverage ─────────────────────────────────────────────────────────────


def test_coverage_lists_all_tickers(store):
    cov = H.coverage_report(store["options_dir"])
    assert set(cov.ticker) == {"SPY", "QQQ"}


def test_coverage_counts_snapshots(store):
    cov = H.coverage_report(store["options_dir"]).set_index("ticker")
    assert cov.loc["SPY", "n_dates"] == 3
    assert cov.loc["QQQ", "n_dates"] == 2


def test_coverage_detects_interior_gap(store):
    """QQQ is missing 2026-01-07, which is a real trading day."""
    cov = H.coverage_report(store["options_dir"]).set_index("ticker")
    assert cov.loc["QQQ", "n_missing"] == 1
    assert "2026-01-07" in cov.loc["QQQ", "missing"]


def test_coverage_complete_ticker_has_no_gaps(store):
    cov = H.coverage_report(store["options_dir"]).set_index("ticker")
    assert cov.loc["SPY", "n_missing"] == 0
    assert cov.loc["SPY", "coverage_pct"] == pytest.approx(100.0)


def test_coverage_empty_store(tmp_path):
    assert H.coverage_report(str(tmp_path / "none")).empty


# ── Trading-day calendar ─────────────────────────────────────────────────


def test_expected_trading_days_excludes_weekends():
    days = H.expected_trading_days(date(2026, 1, 5), date(2026, 1, 11))
    assert all(d.weekday() < 5 for d in days)


def test_expected_trading_days_excludes_holidays():
    """Independence Day 2026 is observed Friday 2026-07-03."""
    days = H.expected_trading_days(date(2026, 7, 1), date(2026, 7, 7))
    assert date(2026, 7, 3) not in days
    assert date(2026, 7, 2) in days


# ── Freshness ────────────────────────────────────────────────────────────


def test_freshness_reports_latest(store):
    f = H.freshness(store["options_dir"], as_of=date(2026, 1, 8))
    assert f["latest"] == date(2026, 1, 8)


def test_freshness_zero_when_current(store):
    f = H.freshness(store["options_dir"], as_of=date(2026, 1, 8))
    assert f["trading_days_stale"] == 0


def test_freshness_counts_stale_sessions(store):
    """Two sessions (01-09 Fri, 01-12 Mon) pass without new data."""
    f = H.freshness(store["options_dir"], as_of=date(2026, 1, 12))
    assert f["trading_days_stale"] == 2


def test_freshness_ignores_weekends(store):
    f = H.freshness(store["options_dir"], as_of=date(2026, 1, 10))  # Saturday
    assert f["trading_days_stale"] == 1  # only Friday 01-09


def test_freshness_flags_lagging_ticker(tmp_path):
    opt = tmp_path / "options"
    for tkr, d in [("SPY", date(2026, 1, 8)), ("QQQ", date(2026, 1, 6))]:
        part = opt / f"ticker={tkr}" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        _chain(tkr, d).to_parquet(part / "chain.parquet", index=False)

    f = H.freshness(str(opt), as_of=date(2026, 1, 8))
    assert f["lagging_tickers"] == ["QQQ"]


def test_freshness_empty_store(tmp_path):
    f = H.freshness(str(tmp_path / "none"))
    assert f["latest"] is None


# ── Quality ──────────────────────────────────────────────────────────────


def test_quality_one_row_per_snapshot(store):
    q = H.quality_report(store["options_dir"])
    assert len(q) == 5  # 3 SPY + 2 QQQ


def test_quality_counts_rows_and_expiries(store):
    q = H.quality_report(store["options_dir"], tickers=["SPY"])
    assert (q.n_rows == 180).all()      # 30 strikes x 6 expiries
    assert (q.n_expiries == 6).all()


def test_quality_flags_thin_snapshot(tmp_path):
    opt = tmp_path / "options"
    part = opt / "ticker=XLF" / "date=2026-01-06"
    part.mkdir(parents=True)
    _chain("XLF", date(2026, 1, 6), n_strikes=2, n_expiries=2).to_parquet(
        part / "chain.parquet", index=False)

    q = H.quality_report(str(opt))
    assert bool(q.thin.iloc[0])
    assert bool(q.few_expiries.iloc[0])


def test_quality_detects_crossed_quotes(tmp_path):
    opt = tmp_path / "options"
    part = opt / "ticker=SPY" / "date=2026-01-06"
    part.mkdir(parents=True)
    _chain("SPY", date(2026, 1, 6), crossed=True).to_parquet(
        part / "chain.parquet", index=False)

    q = H.quality_report(str(opt))
    assert q.crossed_frac.iloc[0] == pytest.approx(1.0)


def test_quality_reports_median_iv(store):
    q = H.quality_report(store["options_dir"], tickers=["SPY"])
    assert q.iv_median.iloc[0] == pytest.approx(0.2)


def test_quality_empty_store(tmp_path):
    assert H.quality_report(str(tmp_path / "none")).empty


# ── Surface status ───────────────────────────────────────────────────────


def test_surface_status_reports_unbuilt(store):
    s = H.surface_status(str(store["root"] / "surfaces"), store["options_dir"])
    s = s.set_index("ticker")
    assert s.loc["SPY", "n_surfaces"] == 0
    assert s.loc["SPY", "n_unbuilt"] == 3


def test_surface_status_counts_built(store):
    surf = store["root"] / "surfaces"
    for d in store["sessions"]:
        p = surf / "ticker=SPY" / f"date={d.isoformat()}"
        p.mkdir(parents=True)
        pd.DataFrame({"log_moneyness": [0.0], "tenor": [0.25],
                      "implied_vol": [0.2]}).to_parquet(p / "surface.parquet")

    s = H.surface_status(str(surf), store["options_dir"]).set_index("ticker")
    assert s.loc["SPY", "n_surfaces"] == 3
    assert s.loc["SPY", "n_unbuilt"] == 0


# ── Aux stores ───────────────────────────────────────────────────────────


def test_rates_status_missing(tmp_path):
    assert H.rates_status(str(tmp_path / "none"))["n_dates"] == 0


def test_rates_status_present(tmp_path):
    from src.data.rates import save_rates_history

    rd = tmp_path / "rates"
    save_rates_history(
        pd.DataFrame({"3M": [0.04]}, index=pd.to_datetime(["2026-01-06"])),
        rates_dir=str(rd),
    )
    st = H.rates_status(str(rd))
    assert st["n_dates"] == 1 and st["tenors"] == ["3M"]


def test_vix_status_missing(tmp_path):
    d = tmp_path / "vix"
    d.mkdir()
    assert H.vix_status(str(d))["n_dates"] == 0


# ── Combined report ──────────────────────────────────────────────────────


def test_full_report_renders(store):
    rep = H.full_report(
        options_dir=store["options_dir"],
        vix_dir=str(store["root"] / "vix"),
        rates_dir=str(store["root"] / "rates"),
        surfaces_dir=str(store["root"] / "surfaces"),
        as_of=date(2026, 1, 8),
    )
    text = rep.render()
    for section in ("DATA HEALTH REPORT", "Freshness", "Coverage",
                    "Quality", "Surfaces", "VIX family", "Risk-free rates"):
        assert section in text


def test_full_report_flags_missing_rates(store):
    rep = H.full_report(
        options_dir=store["options_dir"],
        vix_dir=str(store["root"] / "vix"),
        rates_dir=str(store["root"] / "nope"),
        surfaces_dir=str(store["root"] / "surfaces"),
        as_of=date(2026, 1, 8),
    )
    assert "MISSING" in rep.render()


# ── Collection timing guard ──────────────────────────────────────────────


def test_collection_timing_guard():
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    from src.data.scheduler import collection_allowed

    ny = ZoneInfo("America/New_York")
    d = date(2026, 10, 2)                                   # a regular Friday session
    assert collection_allowed(d, datetime(2026, 10, 2, 16, 30, tzinfo=ny))[0]
    assert not collection_allowed(d, datetime(2026, 10, 2, 11, 0, tzinfo=ny))[0]   # intraday
    assert not collection_allowed(d, datetime(2026, 10, 2, 16, 5, tzinfo=ny))[0]   # options still open
    assert not collection_allowed(d, datetime(2026, 10, 5, 8, 0, tzinfo=ny))[0]    # next-morning catch-up
    half = date(2026, 11, 27)                               # day after Thanksgiving: 13:00 close
    assert collection_allowed(half, datetime(2026, 11, 27, 13, 30, tzinfo=ny))[0]


def test_preclose_window():
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    from src.data.scheduler import preclose_allowed

    ny = ZoneInfo("America/New_York")
    d = date(2026, 10, 2)
    assert preclose_allowed(d, datetime(2026, 10, 2, 15, 45, tzinfo=ny))[0]
    assert not preclose_allowed(d, datetime(2026, 10, 2, 15, 0, tzinfo=ny))[0]
    assert not preclose_allowed(d, datetime(2026, 10, 2, 16, 0, tzinfo=ny))[0]    # quotes pulled
    assert not preclose_allowed(d, datetime(2026, 10, 5, 15, 45, tzinfo=ny))[0]   # another day
    half = date(2026, 11, 27)                                                     # 13:00 close
    assert preclose_allowed(half, datetime(2026, 11, 27, 12, 45, tzinfo=ny))[0]


def test_daily_run_keeps_preclose_chains(tmp_path, monkeypatch):
    """Tickers already collected before the close are not re-scraped at 16:30;
    the others are, and a pre-close ticker with no chain yet still is."""
    from datetime import date

    import src.data.scraper as scraper
    from src.data.scheduler import run_collection
    from src.data.schema import ScraperConfig

    d = date(2026, 10, 2)
    opt = tmp_path / "options"
    (opt / "ticker=XLV" / f"date={d}").mkdir(parents=True)
    scraped = []
    monkeypatch.setattr(scraper, "scrape_all",
                        lambda tickers, **kw: scraped.extend(tickers) or pd.DataFrame())
    cfg = ScraperConfig(tickers=["SPY", "XLV", "HYG"], preclose_tickers=["XLV", "HYG"],
                        output_dir=str(opt), collect_vix=False, collect_rates=False,
                        collect_macro=False, collect_earnings=False, collect_underlying=False,
                        build_surfaces=False, build_features=False)
    run_collection(cfg, as_of=d, force=True)
    assert scraped == ["SPY", "HYG"]
