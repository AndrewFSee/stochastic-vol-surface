"""Forward labels aligned to the feature table, for training downstream models.

``training_frame()`` returns feature rows (historical and live tables, live
winning on overlap) plus forward-looking labels over ``(t, t+h]`` trading
sessions:

* ``label_rv_<h>``  — realised vol over the next *h* sessions,
  ``sqrt(252/h · Σ r²)`` from close-to-close log returns (what a variance
  swap pays);
* ``label_ret_<h>`` — log return over the next *h* sessions;
* ``label_atm30_chg_<h>`` — change in ``atm_30d`` from *t* to the feature row
  *h* sessions later (NaN when that row is missing).

Labels look into the future by construction: never feed them back in as
features, and split train/test by date with a gap of at least *h* sessions
(overlapping windows leak otherwise).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_PATHS = (
    "data/historical/features/surface_features.parquet",
    "data/features/surface_features.parquet",
)


def add_labels(features: pd.DataFrame, prices: pd.DataFrame,
               horizons: Sequence[int] = (5, 21)) -> pd.DataFrame:
    """Add ``label_*`` columns to *features* (``ticker, date`` rows).

    *prices* is the underlying store frame indexed by ``(date, ticker)``.
    """
    from src.features.table import trading_calendar

    out = features.copy()
    for h in horizons:
        for c in (f"label_rv_{h}", f"label_ret_{h}", f"label_atm30_chg_{h}"):
            out[c] = np.nan
    if out.empty:
        return out
    cal = trading_calendar(out["date"].min(), out["date"].max() + pd.Timedelta(days=60))
    tickers_with_prices = set(prices.index.get_level_values("ticker")) if not prices.empty else set()

    for t, idx in out.groupby("ticker").groups.items():
        rows = out.loc[idx]
        dates = pd.DatetimeIndex(rows["date"])
        if t in tickers_with_prices:
            p = prices.xs(t, level="ticker").sort_index()
            close = p["adj_close"].fillna(p["close"]) if "adj_close" in p else p["close"]
            logp = np.log(close.dropna())
            r = logp.diff()
            for h in horizons:
                fwd_sq = (r ** 2)[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
                out.loc[idx, f"label_rv_{h}"] = np.sqrt(TRADING_DAYS / h * fwd_sq).reindex(dates).to_numpy()
                out.loc[idx, f"label_ret_{h}"] = (logp.shift(-h) - logp).reindex(dates).to_numpy()
        if "atm_30d" in rows:
            atm = pd.Series(rows["atm_30d"].to_numpy(), index=dates)
            pos = cal.searchsorted(dates)
            for h in horizons:
                ahead = cal[np.minimum(pos + h, len(cal) - 1)]
                ahead = ahead.where(pos + h < len(cal))
                out.loc[idx, f"label_atm30_chg_{h}"] = (
                    atm.reindex(ahead).to_numpy() - atm.to_numpy())
    return out


def training_frame(
    tickers: Optional[Sequence[str]] = None,
    *,
    horizons: Sequence[int] = (5, 21),
    start: Optional[str] = None,
    end: Optional[str] = None,
    paths: Sequence[str] = DEFAULT_PATHS,
    underlying_dir: str = "data/underlying",
) -> pd.DataFrame:
    """Features from every table plus forward labels (see module docstring)."""
    from pathlib import Path

    from src.data.underlying import load_underlying_history

    frames = [pd.read_parquet(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    if tickers:
        df = df[df["ticker"].isin(list(tickers))]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = load_underlying_history(underlying_dir, tickers=sorted(df["ticker"].unique()))
    df = add_labels(df, prices, horizons)
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)
