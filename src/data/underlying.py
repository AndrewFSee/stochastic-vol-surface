"""Daily OHLC history for the underlyings, for realised vol and returns.

Implied vol is only half of most vol features: the variance risk premium,
IV/RV ratios and return-conditioned skew all need the underlying's own price
path.  Unlike option chains, yfinance serves full daily history, so this
store can be backfilled at any time and a missed day is always recoverable.

Storage
-------
``data/underlying/prices.parquet`` — long format, one row per (date, ticker):
``open, high, low, close`` (as traded), ``adj_close`` (dividend- and
split-adjusted, for returns) and ``volume``.

Public API
----------
``fetch_underlying``            – download a date range for several tickers
``save_underlying_history``     – merge into the store (idempotent)
``load_underlying_history``     – read the store back
``backfill_underlying_history`` – fetch + save in one call
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_UNDERLYING_DIR = "data/underlying"
HISTORY_FILENAME = "prices.parquet"
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

_RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}


def history_path(underlying_dir: str = DEFAULT_UNDERLYING_DIR) -> Path:
    """Path of the consolidated price-history Parquet file."""
    return Path(underlying_dir) / HISTORY_FILENAME


def fetch_underlying(
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "1mo",
) -> pd.DataFrame:
    """Download daily OHLC for *tickers* from yfinance.

    Uses ``start``/``end`` (``end`` exclusive, as in yfinance) when *start* is
    given, otherwise the trailing *period*.  Returns a long frame indexed by
    ``(date, ticker)``; tickers that fail are logged and skipped.
    """
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for tkr in tickers:
        try:
            kw = {"start": start, "end": end} if start else {"period": period}
            hist = yf.Ticker(tkr).history(auto_adjust=False, **kw)
        except Exception as exc:
            logger.error("Failed to fetch %s prices: %s", tkr, exc)
            continue
        if hist is None or hist.empty:
            logger.warning("No price history returned for %s", tkr)
            continue

        df = hist.rename(columns=_RENAME)
        missing = [c for c in PRICE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("%s history lacks %s — skipping", tkr, missing)
            continue
        df = df[PRICE_COLUMNS].copy()
        # yfinance indexes daily bars by exchange-local midnight; keep the date.
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()
        df["ticker"] = tkr
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    out = pd.concat(frames)
    out.index.name = "date"
    return out.reset_index().set_index(["date", "ticker"]).sort_index()


def save_underlying_history(
    df: pd.DataFrame,
    underlying_dir: str = DEFAULT_UNDERLYING_DIR,
) -> Path:
    """Merge *df* into the stored history and write it.

    New non-null values replace stored ones for the same (date, ticker), so a
    re-fetch can correct a provisional bar; nothing already stored is lost.
    """
    out = history_path(underlying_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        merged = df.combine_first(_read(out))
    else:
        merged = df.copy()
    merged = merged[PRICE_COLUMNS].sort_index()
    merged.to_parquet(out, engine="pyarrow")
    logger.info("Saved underlying history: %d rows, %d tickers -> %s",
                len(merged), merged.index.get_level_values("ticker").nunique(), out)
    return out


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["date", "ticker"])
    return df


def load_underlying_history(
    underlying_dir: str = DEFAULT_UNDERLYING_DIR,
    tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load the stored history (indexed by ``(date, ticker)``), or empty."""
    path = history_path(underlying_dir)
    if not path.exists():
        logger.warning("No underlying price history at %s", path)
        return pd.DataFrame(columns=PRICE_COLUMNS)
    df = _read(path).sort_index()
    if tickers is not None:
        df = df[df.index.get_level_values("ticker").isin(list(tickers))]
    return df


def backfill_underlying_history(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    underlying_dir: str = DEFAULT_UNDERLYING_DIR,
) -> pd.DataFrame:
    """Download ``[start, end)`` for *tickers* and merge it into the store."""
    df = fetch_underlying(tickers, start=start, end=end)
    if df.empty:
        logger.error("yfinance returned no prices for %s", list(tickers))
        return df
    save_underlying_history(df, underlying_dir)
    return df
