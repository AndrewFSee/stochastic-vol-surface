"""Tests for the yfinance options chain scraper."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_chain() -> MagicMock:
    """Create a mock yfinance option chain with minimal required columns."""
    calls = pd.DataFrame({
        "contractSymbol": ["SPY230120C00400000"],
        "strike": [400.0],
        "lastPrice": [5.0],
        "bid": [4.8],
        "ask": [5.2],
        "volume": [1000],
        "openInterest": [5000],
        "impliedVolatility": [0.20],
        "inTheMoney": [False],
        "contractSize": ["REGULAR"],
        "currency": ["USD"],
    })
    puts = pd.DataFrame({
        "contractSymbol": ["SPY230120P00390000"],
        "strike": [390.0],
        "lastPrice": [4.0],
        "bid": [3.8],
        "ask": [4.2],
        "volume": [800],
        "openInterest": [4000],
        "impliedVolatility": [0.22],
        "inTheMoney": [False],
        "contractSize": ["REGULAR"],
        "currency": ["USD"],
    })
    mock_chain = MagicMock()
    mock_chain.calls = calls
    mock_chain.puts = puts
    return mock_chain


def test_scrape_options_chain_returns_dataframe():
    """scrape_options_chain should return a DataFrame with required columns."""
    from src.data.scraper import scrape_options_chain

    with patch("src.data.scraper.yf.Ticker") as mock_ticker_cls:
        mock_tkr = MagicMock()
        mock_tkr.options = ["2025-06-20"]
        mock_tkr.option_chain.return_value = _make_mock_chain()
        mock_tkr.fast_info.last_price = 450.0
        mock_ticker_cls.return_value = mock_tkr

        from datetime import date
        df = scrape_options_chain("SPY", as_of=date(2025, 1, 15))

    assert isinstance(df, pd.DataFrame)


def test_scrape_options_chain_required_columns():
    """Returned DataFrame should contain expected schema columns."""
    from src.data.scraper import scrape_options_chain

    with patch("src.data.scraper.yf.Ticker") as mock_ticker_cls:
        mock_tkr = MagicMock()
        mock_tkr.options = ["2025-06-20"]
        mock_tkr.option_chain.return_value = _make_mock_chain()
        mock_tkr.fast_info.last_price = 450.0
        mock_ticker_cls.return_value = mock_tkr

        from datetime import date
        df = scrape_options_chain("SPY", as_of=date(2025, 1, 15))

    required_cols = [
        "ticker", "as_of_date", "expiry_date", "days_to_expiry",
        "option_type", "strike", "bid", "ask",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_scrape_options_chain_open_interest_filter():
    """min_open_interest filter should remove low-OI contracts."""
    from src.data.scraper import scrape_options_chain

    with patch("src.data.scraper.yf.Ticker") as mock_ticker_cls:
        mock_tkr = MagicMock()
        mock_tkr.options = ["2025-06-20"]
        mock_chain = _make_mock_chain()
        # Set very low OI on both legs
        mock_chain.calls["openInterest"] = 1
        mock_chain.puts["openInterest"] = 1
        mock_tkr.option_chain.return_value = mock_chain
        mock_tkr.fast_info.last_price = 450.0
        mock_ticker_cls.return_value = mock_tkr

        from datetime import date
        df = scrape_options_chain("SPY", as_of=date(2025, 1, 15), min_open_interest=100)

    assert df.empty or (df["open_interest"] >= 100).all()


def test_scrape_empty_when_no_expirations():
    """Empty DataFrame returned when ticker has no option expirations."""
    from src.data.scraper import scrape_options_chain

    with patch("src.data.scraper.yf.Ticker") as mock_ticker_cls:
        mock_tkr = MagicMock()
        mock_tkr.options = []
        mock_tkr.fast_info.last_price = 100.0
        mock_ticker_cls.return_value = mock_tkr

        from datetime import date
        df = scrape_options_chain("XYZ", as_of=date(2025, 1, 15))

    assert df.empty
