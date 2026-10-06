"""Forecasting dataset: realised-variance inputs, forward targets, surface features.

One row per trading date *t*, using only information known at the close of
*t* (the feature table's own timing contract):

* ``rv_d, rv_w, rv_m`` — HAR inputs: annualised daily variance estimates
  averaged over 1, 5 and 22 days.  Daily variance is Garman-Klass from
  dividend-adjusted OHLC plus the overnight gap, which is several times less
  noisy than a squared close-to-close return.
* ``ret`` — the close-to-close log return of *t* (GARCH input).
* ``y_<h>`` — the target: annualised close-to-close realised variance over
  the next *h* trading days, ``252/h · Σ_{i=1..h} r²_{t+i}``.  This is what a
  variance swap pays, so implied variance is scored on what it prices.  NaN
  until all *h* returns exist.
* surface and market features from the feature tables (NaN where no surface
  exists, e.g. 2024–Feb 2026 for SPY).
* earnings (single stocks; see :mod:`src.features.earnings`):
  ``ev_day`` (*t* is an earnings session), ``ev_next`` (next one after *t*),
  ``ev_in_<h>`` (one falls in ``(t, t+h]``), ``ev_hist_var`` (historical
  excess event-day variance), and ex-event versions of the HAR inputs
  (``rv_*_ex``, event days left out) and targets (``y_<h>_ex``, the
  annualised mean squared return over the window's non-event days).  For
  tickers without earnings these equal the plain columns.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (5, 21)

#: Feature-table columns carried into the dataset.
FEATURE_COLUMNS = [
    "atm_7d", "atm_30d", "atm_91d", "atm_365d", "vs_30d", "vs_91d",
    "rr25_30d", "bf25_30d", "rr25_91d", "atm_skew_30d", "atm_curv_30d",
    "ts_7_30", "ts_30_91", "ts_30_365", "vrp_30d",
    "mkt_vix", "mkt_vix3m", "mkt_vix9d", "mkt_vvix", "mkt_skew",
    "earn_implied_move",
]

DEFAULT_FEATURE_PATHS = (
    "data/historical/features/surface_features.parquet",
    "data/features/surface_features.parquet",
)


def daily_variance(prices: pd.DataFrame) -> pd.Series:
    """Annualised Garman-Klass daily variance including the overnight gap.

    ``σ² = ln(O/C₋₁)² + ½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)²`` on OHLC scaled
    by the day's adjustment factor, so dividends do not show up as variance.
    """
    p = prices.sort_index()
    f = (p["adj_close"] / p["close"]).ffill().fillna(1.0)
    o, h, l, c = (np.log(p[col] * f) for col in ("open", "high", "low", "close"))
    overnight = o - c.shift(1)
    gk = 0.5 * (h - l) ** 2 - (2 * math.log(2) - 1) * (c - o) ** 2
    return TRADING_DAYS * (overnight ** 2 + gk.clip(lower=0))


def forward_realised_variance(ret: pd.Series, h: int) -> pd.Series:
    """``252/h · Σ_{i=1..h} r²_{t+i}``, NaN where the window is incomplete."""
    sq = ret ** 2
    fwd = sq[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return TRADING_DAYS / h * fwd


def load_features(
    ticker: str,
    paths: Iterable[str] = DEFAULT_FEATURE_PATHS,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """The ticker's rows from every available feature table, newest table winning."""
    from pathlib import Path

    frames = []
    for p in paths:
        if Path(p).exists():
            f = pd.read_parquet(p)
            frames.append(f[f["ticker"] == ticker].set_index("date"))
    if not frames:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[[c for c in columns if c in out.columns]]


def build_dataset(
    ticker: str = "SPY",
    *,
    underlying_dir: str = "data/underlying",
    feature_paths: Sequence[str] = DEFAULT_FEATURE_PATHS,
    horizons: Sequence[int] = HORIZONS,
    events_dir: str = "data/events",
) -> pd.DataFrame:
    """Assemble the modelling frame for *ticker* (see module docstring)."""
    from src.data.underlying import load_underlying_history

    from src.features.table import fill_closes_from_snapshots

    prices = load_underlying_history(underlying_dir, tickers=[ticker])
    if prices.empty:
        raise ValueError(f"no price history for {ticker}; run scripts/backfill_underlying.py")
    # At 16:30 ET Yahoo's bar for the day has no close yet; the option
    # snapshot's spot is the close (see fill_closes_from_snapshots).
    spots = load_features(ticker, feature_paths, columns=["spot"])["spot"].dropna()
    if not spots.empty:
        spots.index = pd.MultiIndex.from_arrays([spots.index, [ticker] * len(spots)],
                                                names=["date", "ticker"])
        prices = fill_closes_from_snapshots(prices, spots)
    p = prices.xs(ticker, level="ticker").sort_index()
    p = p[p["close"].notna() & p["open"].notna()]

    ds = pd.DataFrame(index=p.index)
    ds["ret"] = np.log(p["adj_close"]).diff()
    dv = daily_variance(p)
    ds["rv_d"] = dv
    ds["rv_w"] = dv.rolling(5, min_periods=5).mean()
    ds["rv_m"] = dv.rolling(22, min_periods=22).mean()
    for h in horizons:
        ds[f"y_{h}"] = forward_realised_variance(ds["ret"], h)
    _add_events(ds, dv, p, ticker, events_dir, horizons)

    ds = ds.join(load_features(ticker, feature_paths), how="left")
    ds.index.name = "date"
    return ds.iloc[1:]


def _add_events(ds: pd.DataFrame, dv: pd.Series, p: pd.DataFrame, ticker: str,
                events_dir: str, horizons: Sequence[int]) -> None:
    """Earnings columns, in place (see module docstring)."""
    from src.data.events import event_sessions, load_earnings
    from src.features.earnings import calendar_columns, historical_event_variance
    from src.features.table import trading_calendar

    earnings = load_earnings(events_dir)
    earnings = earnings[earnings["ticker"] == ticker] if not earnings.empty else earnings
    events = pd.DatetimeIndex([])
    sessions = None
    if not earnings.empty:
        sessions = trading_calendar(min(ds.index.min(), pd.Timestamp("2000-01-01")),
                                    ds.index.max() + pd.Timedelta(days=400))
        events = event_sessions(earnings, sessions).get(ticker, events)

    is_ev = ds.index.isin(events)
    ds["ev_day"] = is_ev
    dv_ex = dv.where(~is_ev)
    ds["rv_d_ex"] = dv_ex.ffill(limit=1)
    ds["rv_w_ex"] = dv_ex.rolling(5, min_periods=4).mean()
    ds["rv_m_ex"] = dv_ex.rolling(22, min_periods=18).mean()

    sq = ds["ret"] ** 2
    sq_ex = sq.where(~is_ev).fillna(0.0)
    n_ex = pd.Series((~is_ev).astype(float), index=ds.index)
    for h in horizons:
        def fwd(x: pd.Series) -> pd.Series:
            return x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
        complete = fwd(sq).notna()
        ds[f"y_{h}_ex"] = (TRADING_DAYS * fwd(sq_ex) / fwd(n_ex)).where(complete)

    if len(events) == 0:
        ds["ev_next"] = pd.NaT
        ds["ev_hist_var"] = np.nan
        for h in horizons:
            ds[f"ev_in_{h}"] = False
        return
    cal = calendar_columns(ds.index, events, sessions)
    ds["ev_next"] = cal["earn_next_date"]
    for h in horizons:
        ds[f"ev_in_{h}"] = (cal["earn_days_to"] <= h).fillna(False).astype(bool)
    close = p["adj_close"].fillna(p["close"])
    ds["ev_hist_var"] = historical_event_variance(close, events).reindex(ds.index)
