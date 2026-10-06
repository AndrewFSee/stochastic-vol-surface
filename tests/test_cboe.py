"""CBOE delayed-quotes fallback: parsing and when it replaces yfinance."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.data import cboe


def _payload(n_good=4, n_dead=0):
    opts = []
    for i in range(n_good):
        opts.append({"option": f"SPY261218{'CP'[i % 2]}{(800 + i) * 1000:08d}", "bid": 3.5,
                     "ask": 3.6, "iv": 0.12, "open_interest": 5, "volume": 1,
                     "last_trade_price": 3.55})
    for i in range(n_dead):
        opts.append({"option": f"SPY261218C{(900 + i) * 1000:08d}", "bid": 0.0, "ask": 0.0,
                     "iv": 0.0, "open_interest": 0, "volume": 0, "last_trade_price": 0})
    return {"data": {"symbol": "SPY", "current_price": 778.28, "options": opts}}


def test_occ_symbols_parse_from_the_right():
    got = cboe.parse_occ(pd.Series(["SPY261218C00824000", "BRKB270115P00412500"]))
    assert got["expiration"].tolist() == [pd.Timestamp("2026-12-18"), pd.Timestamp("2027-01-15")]
    assert got["option_type"].tolist() == ["call", "put"]
    assert got["strike"].tolist() == [824.0, 412.5]


def test_normalise_gives_canonical_columns():
    df = cboe.normalise(_payload(n_good=2, n_dead=1), "SPY", date(2026, 10, 6))
    assert len(df) == 3 and (df["source"] == "cboe").all()
    assert (df["underlying_price"] == 778.28).all()
    assert df["implied_volatility_market"].isna().sum() == 1      # iv 0 means no IV
    assert df["T"].iloc[0] == (pd.Timestamp("2026-12-18") - pd.Timestamp("2026-10-06")).days / 365
    assert cboe.two_sided(df) == 2


def test_fallback_replaces_only_empty_or_thin_chains(monkeypatch):
    from src.data import scraper

    good_cboe = cboe.normalise(_payload(n_good=10), "XLV", date(2026, 10, 6))
    monkeypatch.setattr(cboe, "fetch_cboe_chain", lambda t, d: good_cboe)

    healthy = cboe.normalise(_payload(n_good=4), "XLV", date(2026, 10, 6)).assign(source="yfinance")
    assert (scraper._with_fallback("XLV", date(2026, 10, 6), healthy)["source"] == "yfinance").all()

    thin = cboe.normalise(_payload(n_good=1, n_dead=3), "XLV", date(2026, 10, 6)).assign(source="yfinance")
    assert (scraper._with_fallback("XLV", date(2026, 10, 6), thin)["source"] == "cboe").all()
    assert (scraper._with_fallback("XLV", date(2026, 10, 6), pd.DataFrame())["source"] == "cboe").all()

    monkeypatch.setattr(cboe, "fetch_cboe_chain", lambda t, d: pd.DataFrame())   # CBOE down too
    assert (scraper._with_fallback("XLV", date(2026, 10, 6), thin)["source"] == "yfinance").all()
