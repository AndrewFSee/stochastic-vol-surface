"""Earnings dates for the single-stock tickers (yfinance).

An earnings release moves a stock in one session, by far more than a normal
day, and option prices carry that jump.  The forecasts and features need to
know which session each release lands on:

* released before the open (or during the day)  → that day's session;
* released after the close (16:00 ET or later)  → the next session.

Yahoo's history reaches back to about 2002 and includes the next scheduled
date.  Dates that were still estimates at the time are stored as the actual
date, so a row long before a release may see a date that was not confirmed
yet; within the last few weeks before a release the date is almost always
public.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_DIR = "data/events"
EARNINGS_FILENAME = "earnings.parquet"

#: Releases at or after this hour (ET) move the next session.
AFTER_CLOSE_HOUR = 12


def earnings_path(events_dir: str = DEFAULT_EVENTS_DIR) -> Path:
    return Path(events_dir) / EARNINGS_FILENAME


def fetch_earnings(tickers: Iterable[str], limit: int = 100) -> pd.DataFrame:
    """Past and scheduled earnings releases for *tickers*.

    Funds and indices have none and are skipped.  Returns columns ``ticker,
    announced`` (tz-aware, America/New_York), ``eps_estimate,
    eps_reported, surprise_pct``.
    """
    import yfinance as yf

    frames = []
    for t in tickers:
        try:
            d = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception as exc:
            logger.warning("Earnings dates for %s failed: %s", t, exc)
            continue
        if d is None or d.empty:
            continue
        d = d.reset_index().rename(columns={
            "Earnings Date": "announced", "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_reported", "Surprise(%)": "surprise_pct"})
        d["ticker"] = t
        frames.append(d[["ticker", "announced", "eps_estimate", "eps_reported", "surprise_pct"]])
    if not frames:
        return pd.DataFrame(columns=["ticker", "announced", "eps_estimate",
                                     "eps_reported", "surprise_pct"])
    out = pd.concat(frames, ignore_index=True)
    out["announced"] = pd.to_datetime(out["announced"], utc=True).dt.tz_convert("America/New_York")
    return out


def save_earnings(df: pd.DataFrame, events_dir: str = DEFAULT_EVENTS_DIR) -> Path:
    """Store *df*, replacing every ticker it covers (a fetch returns the full
    history, so a rescheduled date replaces the old one); other tickers stay."""
    out = earnings_path(events_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old[~old["ticker"].isin(df["ticker"].unique())], df], ignore_index=True)
    df = df.sort_values(["ticker", "announced"]).reset_index(drop=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Saved %d earnings dates for %d tickers -> %s",
                len(df), df["ticker"].nunique(), out)
    return out


def load_earnings(events_dir: str = DEFAULT_EVENTS_DIR) -> pd.DataFrame:
    p = earnings_path(events_dir)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "announced"])
    return pd.read_parquet(p)


def event_sessions(earnings: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """The session each release moves, per ticker, on the calendar *sessions*.

    Releases outside the calendar's range are dropped.
    """
    if earnings.empty:
        return {}
    sessions = pd.DatetimeIndex(sessions).normalize().sort_values()
    out = {}
    for t, g in earnings.groupby("ticker"):
        ts = pd.DatetimeIndex(g["announced"])
        day = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
        after = ts.hour >= AFTER_CLOSE_HOUR
        pos = sessions.searchsorted(day, side="left")
        pos = np.where(after & (pos < len(sessions)) & (sessions[np.minimum(pos, len(sessions) - 1)] == day),
                       pos + 1, pos)
        ok = (pos < len(sessions)) & (day >= sessions[0])
        out[t] = pd.DatetimeIndex(sorted(set(sessions[pos[ok]])))
    return out
