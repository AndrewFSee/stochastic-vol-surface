"""Download US Treasury risk-free rates from the FRED API.

Requires environment variable ``FRED_API_KEY`` (free registration at
https://fred.stlouisfed.org/docs/api/api_key.html).

Series used
-----------
DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RATE_SERIES: dict[str, str] = {
    "1M":  "DGS1MO",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "3Y":  "DGS3",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
}


def fetch_rates(
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns for each tenor.

    Values are annualised yields expressed as decimals (e.g. 0.045 = 4.5%).
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=key)
    frames: dict[str, pd.Series] = {}

    for label, series_id in RATE_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            if data is not None and not data.empty:
                frames[label] = data / 100.0  # Convert from percent to decimal
        except Exception as exc:
            logger.warning("Failed to fetch %s (%s): %s", label, series_id, exc)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result.sort_index()


def get_risk_free_rate(
    tenor: str = "3M",
    api_key: Optional[str] = None,
) -> float:
    """Return the latest available risk-free rate for a given tenor.

    Parameters
    ----------
    tenor : str
        One of "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y".
    """
    df = fetch_rates(api_key=api_key)
    if df.empty or tenor not in df.columns:
        logger.warning("Using fallback risk-free rate 0.05")
        return 0.05
    return float(df[tenor].dropna().iloc[-1])


# ────────────────────────────────────────────────────────────────────────────
# Persistent rate history  (single consolidated file, not per-day snapshots)
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_RATES_DIR = "data/rates"
HISTORY_FILENAME = "rates_history.parquet"

# Tenor labels → years, used for term interpolation.
TENOR_YEARS: dict[str, float] = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0,
    "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0,
}

FALLBACK_RATE = 0.05


def history_path(rates_dir: str = DEFAULT_RATES_DIR) -> Path:
    """Return the path of the consolidated rate-history Parquet file."""
    return Path(rates_dir) / HISTORY_FILENAME


def save_rates_history(
    df: pd.DataFrame,
    rates_dir: str = DEFAULT_RATES_DIR,
) -> Path:
    """Merge *df* into the consolidated rate history and write it to disk.

    Existing rows are kept unless *df* supplies a non-null replacement for the
    same (date, tenor) cell, so re-running a backfill is idempotent and can
    only add information.
    """
    out = history_path(rates_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        existing = pd.read_parquet(out)
        existing.index = pd.to_datetime(existing.index)
        # combine_first prefers `df`, falling back to prior values.
        merged = df.combine_first(existing)
    else:
        merged = df.copy()

    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"
    merged = merged.sort_index()
    merged.to_parquet(out, engine="pyarrow")
    logger.info("Saved rate history: %d dates x %d tenors -> %s",
                len(merged), merged.shape[1], out)
    return out


def load_rates_history(rates_dir: str = DEFAULT_RATES_DIR) -> pd.DataFrame:
    """Load the consolidated rate history, or an empty frame if absent."""
    path = history_path(rates_dir)
    if not path.exists():
        logger.warning("No rate history at %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def backfill_rates_history(
    start: str,
    end: Optional[str] = None,
    rates_dir: str = DEFAULT_RATES_DIR,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download the full FRED curve for [start, end] and persist it.

    FRED serves complete history, so a missed daily snapshot is always
    recoverable after the fact — unlike the options chains.
    """
    df = fetch_rates(start=start, end=end, api_key=api_key)
    if df.empty:
        logger.error("FRED returned no data for %s..%s", start, end)
        return df
    save_rates_history(df, rates_dir=rates_dir)
    return df


def get_rate_curve(
    as_of,
    rates_dir: str = DEFAULT_RATES_DIR,
    history: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Return the yield curve on *as_of*, as ``{tenor_label: rate}``.

    Uses the most recent curve at or before *as_of* (Treasury series are not
    published on market holidays), forward-filling individual missing tenors.
    """
    hist = load_rates_history(rates_dir) if history is None else history
    if hist.empty:
        return {}

    ts = pd.Timestamp(as_of)
    upto = hist.loc[hist.index <= ts]
    if upto.empty:
        return {}

    # Forward-fill so a single missing tenor falls back to its last print.
    row = upto.ffill().iloc[-1]
    return {k: float(v) for k, v in row.items() if pd.notna(v)}


def get_rate_for_tenor(
    as_of,
    T: float,
    rates_dir: str = DEFAULT_RATES_DIR,
    history: Optional[pd.DataFrame] = None,
) -> float:
    """Return the risk-free rate for maturity *T* (years) on date *as_of*.

    Log-linearly interpolates across the curve's tenor points and clamps to the
    nearest available tenor outside the curve's range.  Falls back to
    :data:`FALLBACK_RATE` when no curve is available.
    """
    curve = get_rate_curve(as_of, rates_dir=rates_dir, history=history)
    if not curve:
        return FALLBACK_RATE

    pts = sorted(
        (TENOR_YEARS[k], v) for k, v in curve.items() if k in TENOR_YEARS
    )
    if not pts:
        return FALLBACK_RATE

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) == 1:
        return float(ys[0])

    import numpy as np

    # np.interp clamps to the endpoints outside [xs[0], xs[-1]], which is the
    # desired behaviour for very short / very long dated options.
    return float(np.interp(max(float(T), 1e-6), xs, ys))


def get_rate_term_structure(
    as_of,
    tenors,
    rates_dir: str = DEFAULT_RATES_DIR,
    history: Optional[pd.DataFrame] = None,
) -> "pd.Series":
    """Return interpolated rates for an iterable of maturities *tenors*."""
    hist = load_rates_history(rates_dir) if history is None else history
    vals = [
        get_rate_for_tenor(as_of, float(T), rates_dir=rates_dir, history=hist)
        for T in tenors
    ]
    return pd.Series(vals, index=list(tenors), name="rate")
