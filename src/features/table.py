"""The feature table: one point-in-time row per (ticker, trading date).

This is the project's main output.  The dashboard reads it, and so should
any downstream model.  It is rebuilt in full from the surface store, the
underlying price history and the VIX history, so it is deterministic and
never holds stale derived columns.

Timing
------
A row dated *t* uses only information available at about 16:30 ET on *t*:
that day's option snapshot, the underlying's close and the VIX-family
closes.  To predict anything over ``(t, t+h]``, use row *t*.  Every rolling
statistic is trailing and includes *t* itself, never later rows.

Columns
-------
``ticker, date`` keys; surface features (:mod:`.surface_features`);
realised features (:mod:`.realized`); ``vrp_*`` premia; ``mkt_*`` VIX
family; ``<col>_d1`` / ``_d5`` changes, ``_z63`` z-scores and ``_pct252``
percentile ranks for :data:`DYNAMIC_COLUMNS`; and provenance
(``builder``, ``feature_version``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_VERSION = "features/1"
DEFAULT_FEATURES_PATH = "data/features/surface_features.parquet"

#: Columns that get changes, z-scores and percentile ranks.
DYNAMIC_COLUMNS = [
    "atm_30d", "atm_91d", "vs_30d", "rr25_30d", "bf25_30d",
    "ts_7_30", "ts_30_91", "vrp_30d",
]
Z_WINDOW = 63
PCT_WINDOW = 252
MIN_PERIODS = 20

_VIX_COLUMNS = {"VIX": "mkt_vix", "VIX3M": "mkt_vix3m", "VIX9D": "mkt_vix9d",
                "VVIX": "mkt_vvix", "SKEW": "mkt_skew"}


# ────────────────────────────────────────────────────────────────────────────
# Building
# ────────────────────────────────────────────────────────────────────────────


def _surface_rows(ticker: str, surfaces_dir: str, start, end) -> list[dict]:
    from src.features.surface_features import surface_features
    from src.surface.batch import load_surfaces

    rows = []
    for d, vs in load_surfaces(ticker, surfaces_dir=surfaces_dir,
                               start=start, end=end).items():
        if not vs.slices:
            continue
        rows.append({
            "ticker": ticker,
            "date": pd.Timestamp(d),
            "spot": float(vs.spot),
            **surface_features(vs.slices, spot=vs.spot),
            "builder": vs.builder,
        })
    return rows


def fill_closes_from_snapshots(prices: pd.DataFrame, spots: pd.Series) -> pd.DataFrame:
    """Use the option snapshot's spot where the daily bar has no close.

    At 16:30 ET Yahoo's bar for the day is still provisional: open, high and
    low are there but close is NaN, so the newest row would always lack
    realised vol.  The snapshot spot is the official close (SPY 2026-10-01:
    763.99 in both), so it fills the gap.  The next day's price fetch
    replaces the provisional bar in the store.

    *spots* is indexed by ``(date, ticker)``.
    """
    spots = spots.dropna()
    if spots.empty:
        return prices
    p = prices.copy() if not prices.empty else pd.DataFrame(
        columns=["open", "high", "low", "close", "adj_close", "volume"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    missing = spots.index.difference(p.index)
    if len(missing):
        p = pd.concat([p, pd.DataFrame(index=missing, columns=p.columns, dtype=float)])
    no_close = p["close"].isna() & p.index.isin(spots.index)
    p.loc[no_close, "close"] = spots.reindex(p.index[no_close]).to_numpy()
    return p.sort_index()


def trading_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """NYSE sessions in ``[start, end]``; weekdays if the calendar is unavailable."""
    try:
        import pandas_market_calendars as mcal

        days = mcal.get_calendar("NYSE").valid_days(start.date(), end.date())
        return pd.DatetimeIndex(days).tz_localize(None).normalize()
    except Exception:
        return pd.bdate_range(start, end)


def add_dynamics(
    df: pd.DataFrame,
    columns: Sequence[str] = DYNAMIC_COLUMNS,
    *,
    z_window: int = Z_WINDOW,
    pct_window: int = PCT_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    """Add trailing changes, z-scores and percentile ranks per ticker.

    Each ticker's series is laid on the full NYSE calendar first, so a missing
    snapshot leaves a gap: ``_d1`` across it is NaN instead of silently
    spanning two days.  Windows include the current row and nothing later.
    """
    if df.empty:
        return df
    cols = [c for c in columns if c in df.columns]
    cal = trading_calendar(df["date"].min(), df["date"].max())
    out = []
    for _tkr, g in df.groupby("ticker", sort=False):
        g = g.set_index("date").sort_index()
        full = g[cols].reindex(cal.union(g.index))
        dyn = {}
        for c in cols:
            x = full[c]
            dyn[f"{c}_d1"] = x.diff(1)
            dyn[f"{c}_d5"] = x.diff(5)
            roll = x.rolling(z_window, min_periods=min_periods)
            dyn[f"{c}_z{z_window}"] = (x - roll.mean()) / roll.std()
            dyn[f"{c}_pct{pct_window}"] = x.rolling(
                pct_window, min_periods=min_periods).rank(pct=True)
        dyn = pd.DataFrame(dyn).reindex(g.index)
        out.append(pd.concat([g, dyn], axis=1).reset_index())
    return pd.concat(out, ignore_index=True)


def build_feature_table(
    tickers: Optional[Iterable[str]] = None,
    *,
    surfaces_dir: str = "data/surfaces",
    underlying_dir: str = "data/underlying",
    vix_dir: str = "data/vix",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Assemble the full feature table from the stores on disk."""
    from src.data.underlying import load_underlying_history
    from src.data.vix_family import load_vix_history
    from src.features.realized import realized_features
    from src.surface.batch import available_tickers

    tickers = list(tickers) if tickers else available_tickers(surfaces_dir)
    rows: list[dict] = []
    for tkr in tickers:
        r = _surface_rows(tkr, surfaces_dir, start, end)
        logger.info("%-6s %d surfaces with fits", tkr, len(r))
        rows.extend(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # ── Realised features (computed on the full price history, then joined,
    #    so windows reach back before the first surface date) ──────────────
    prices = fill_closes_from_snapshots(
        load_underlying_history(underlying_dir, tickers=tickers),
        df.set_index(["date", "ticker"])["spot"],
    )
    if prices.empty:
        logger.warning("No underlying prices — realised-vol and VRP features "
                       "will be NaN. Run scripts/backfill_underlying.py.")
    else:
        rv = []
        for tkr, g in prices.groupby(level="ticker"):
            feats = realized_features(g.droplevel("ticker"))
            feats["ticker"] = tkr
            rv.append(feats.reset_index())
        df = df.merge(pd.concat(rv, ignore_index=True), on=["ticker", "date"], how="left")

    for col in ("rv_cc_21d", "rv_yz_21d"):
        if col not in df.columns:
            df[col] = np.nan
    # Implied minus realised.  The vol-point version uses ATM; the variance
    # version pairs the variance swap with zero-mean realised variance, which
    # is the actual P&L of a variance swap held to expiry.
    df["vrp_30d"] = df["atm_30d"] - df["rv_cc_21d"]
    df["vrp_var_30d"] = df["vs_30d"] ** 2 - df["rv_cc_21d"] ** 2

    # ── Market context ───────────────────────────────────────────────────
    vix = load_vix_history(vix_dir)
    if not vix.empty:
        mkt = vix[[c for c in _VIX_COLUMNS if c in vix.columns]].rename(columns=_VIX_COLUMNS)
        mkt.index = pd.to_datetime(mkt.index).normalize()
        df = df.merge(mkt, left_on="date", right_index=True, how="left")

    df = add_dynamics(df)
    df["feature_version"] = FEATURE_VERSION
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    lead = ["ticker", "date"]
    tail = ["builder", "feature_version"]
    return df[lead + [c for c in df.columns if c not in lead + tail] + tail]


# ────────────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────────────


def save_feature_table(df: pd.DataFrame, path: str = DEFAULT_FEATURES_PATH) -> Path:
    """Write the table to Parquet (replacing any previous build)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Saved feature table: %d rows x %d columns -> %s",
                len(df), df.shape[1], out)
    return out


def load_feature_table(
    path: str = DEFAULT_FEATURES_PATH,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load the feature table, optionally filtered by ticker and date."""
    p = Path(path)
    if not p.exists():
        logger.warning("No feature table at %s — run scripts/build_features.py", p)
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if tickers is not None:
        df = df[df["ticker"].isin(list(tickers))]
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)
