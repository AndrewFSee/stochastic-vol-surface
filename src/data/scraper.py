"""Scrape live options chains from yfinance and normalise into a canonical DataFrame.

Canonical columns
-----------------
ticker, as_of, expiration, strike, option_type, bid, ask, mid, last_price,
volume, open_interest, implied_volatility_market, T, underlying_price
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_yfinance_chain(ticker_str: str, as_of: date) -> pd.DataFrame:
    """Download and normalise every available expiry for *ticker_str*."""
    import yfinance as yf

    tk = yf.Ticker(ticker_str)

    # Current spot price
    info = tk.fast_info
    spot = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
    if spot is None:
        hist = tk.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if len(hist) else np.nan
    spot = float(spot)

    expirations = tk.options  # tuple of 'YYYY-MM-DD' strings
    if not expirations:
        logger.warning("No options expirations for %s", ticker_str)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for exp_str in expirations:
        try:
            chain = tk.option_chain(exp_str)
        except Exception as exc:
            logger.warning("Failed to fetch %s %s: %s", ticker_str, exp_str, exc)
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            if df.empty:
                continue
            tmp = df.copy()
            tmp["option_type"] = opt_type
            tmp["expiration"] = pd.Timestamp(exp_str)
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Rename yfinance columns to canonical names
    rename_map = {
        "lastPrice": "last_price",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_volatility_market",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    # Compute derived fields
    raw["ticker"] = ticker_str
    raw["as_of"] = pd.Timestamp(as_of)
    raw["underlying_price"] = spot

    # Mid price
    if "bid" in raw.columns and "ask" in raw.columns:
        raw["mid"] = (raw["bid"] + raw["ask"]) / 2
    elif "last_price" in raw.columns:
        raw["mid"] = raw["last_price"]

    # Time-to-expiry in years (ACT/365)
    raw["T"] = (raw["expiration"] - raw["as_of"]).dt.days / 365.0

    # Drop very short-dated (< 1 calendar day) and negative T
    raw = raw[raw["T"] > 0].copy()

    # Select canonical columns (keep extras if present)
    canonical = [
        "ticker", "as_of", "expiration", "strike", "option_type",
        "bid", "ask", "mid", "last_price",
        "volume", "open_interest", "implied_volatility_market",
        "T", "underlying_price",
    ]
    cols = [c for c in canonical if c in raw.columns]
    return raw[cols].reset_index(drop=True)


def scrape_all(
    tickers: list[str],
    as_of: Optional[date] = None,
    inter_ticker_delay: float = 1.5,
) -> pd.DataFrame:
    """Scrape options chains for multiple tickers.

    Parameters
    ----------
    tickers : list[str]
        e.g. ["SPY", "QQQ", "AAPL", "MSFT"]
    as_of : date or None
        As-of date stamp; defaults to today.
    inter_ticker_delay : float
        Seconds to sleep between tickers to avoid yfinance rate-limits.

    Returns
    -------
    pd.DataFrame  with canonical columns.
    """
    if as_of is None:
        as_of = date.today()

    frames: list[pd.DataFrame] = []
    for i, tkr in enumerate(tickers):
        if i > 0 and inter_ticker_delay > 0:
            time.sleep(inter_ticker_delay)
        logger.info("Scraping %s (%d/%d) …", tkr, i + 1, len(tickers))
        try:
            df = _parse_yfinance_chain(tkr, as_of)
            if not df.empty:
                frames.append(df)
                logger.info("  → %d rows for %s", len(df), tkr)
            else:
                logger.warning("  → empty chain for %s", tkr)
        except Exception as exc:
            logger.error("  → error scraping %s: %s", tkr, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
