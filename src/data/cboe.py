"""Fallback option chains from CBOE's free delayed-quotes feed.

``https://cdn-api.cboe.com/api/global/delayed_quotes/options/<SYMBOL>.json``
returns every listed contract with bid, ask, volume, open interest and IV,
delayed about 15 minutes.  The daily scrape uses it when yfinance fails or
returns a chain with too few two-sided quotes (see
:func:`src.data.scraper.scrape_all`).  Output uses the scraper's canonical
columns, with ``source = "cboe"``.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
_HEADERS = {"User-Agent": "Mozilla/5.0 (stochastic-vol-surface)"}


def _get(symbol: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(URL.format(symbol=symbol), headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_occ(symbols: pd.Series) -> pd.DataFrame:
    """Split OCC option symbols (``SPY261218C00824000``) into expiration,
    option_type and strike.  The root has variable length, so parse from the
    right: 6-digit date, C/P, strike × 1000 in 8 digits."""
    s = symbols.astype(str)
    return pd.DataFrame({
        "expiration": pd.to_datetime(s.str[-15:-9], format="%y%m%d"),
        "option_type": s.str[-9].map({"C": "call", "P": "put"}),
        "strike": s.str[-8:].astype(int) / 1000.0,
    }, index=symbols.index)


def normalise(payload: dict, ticker: str, as_of: date) -> pd.DataFrame:
    """A CBOE delayed-quotes payload in the scraper's canonical columns."""
    data = payload.get("data") or {}
    opts = pd.DataFrame(data.get("options") or [])
    if opts.empty:
        return pd.DataFrame()
    spot = data.get("current_price") or data.get("close")
    df = parse_occ(opts["option"])
    for c in ("bid", "ask", "volume", "open_interest"):
        df[c] = pd.to_numeric(opts.get(c), errors="coerce")
    df["last_price"] = pd.to_numeric(opts.get("last_trade_price"), errors="coerce")
    iv = pd.to_numeric(opts.get("iv"), errors="coerce")
    df["implied_volatility_market"] = iv.where(iv > 0)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["ticker"] = ticker
    df["as_of"] = pd.Timestamp(as_of)
    df["underlying_price"] = float(spot) if spot else np.nan
    df["T"] = (df["expiration"] - df["as_of"]).dt.days / 365.0
    df["source"] = "cboe"
    df = df[(df["T"] > 0) & df["option_type"].notna()]
    cols = ["ticker", "as_of", "expiration", "strike", "option_type", "bid", "ask", "mid",
            "last_price", "volume", "open_interest", "implied_volatility_market", "T",
            "underlying_price", "source"]
    return df[cols].reset_index(drop=True)


def fetch_cboe_chain(ticker: str, as_of: date) -> pd.DataFrame:
    """Download and normalise *ticker*'s full chain from CBOE (empty on failure)."""
    try:
        return normalise(_get(ticker), ticker, as_of)
    except Exception as exc:
        logger.warning("CBOE chain for %s failed: %s", ticker, exc)
        return pd.DataFrame()


def two_sided(df: pd.DataFrame) -> int:
    """Contracts with both a positive bid and a positive ask."""
    if df.empty or "bid" not in df or "ask" not in df:
        return 0
    return int(((df["bid"] > 0) & (df["ask"] > 0)).sum())
