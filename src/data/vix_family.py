"""Download VIX-family indices from yfinance: VIX, VIX3M, VIX9D, SKEW, VVIX."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

VIX_TICKERS: dict[str, str] = {
    "VIX":   "^VIX",
    "VIX3M": "^VIX3M",
    "VIX9D": "^VIX9D",
    "SKEW":  "^SKEW",
    "VVIX":  "^VVIX",
}


def fetch_vix_family(
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns VIX, VIX3M, VIX9D, SKEW, VVIX.

    Parameters
    ----------
    start, end : str "YYYY-MM-DD" or None
        If both are provided, fetches that range.  Otherwise uses *period*.
    period : str
        yfinance period string, e.g. "5y", "1y", "6mo".
    """
    import yfinance as yf

    frames: dict[str, pd.Series] = {}

    for name, yf_ticker in VIX_TICKERS.items():
        try:
            if start and end:
                hist = yf.download(yf_ticker, start=start, end=end, progress=False)
            else:
                hist = yf.download(yf_ticker, period=period, progress=False)

            if hist.empty:
                logger.warning("No data returned for %s (%s)", name, yf_ticker)
                continue

            # Use Close (or Adj Close) price
            col = "Adj Close" if "Adj Close" in hist.columns else "Close"
            # yfinance may return multi-level columns for single ticker
            series = hist[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            frames[name] = series.rename(name)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", name, exc)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames.values(), axis=1)
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result.sort_index()


def fetch_latest_vix() -> dict[str, float]:
    """Return the most recent close for each VIX-family index."""
    df = fetch_vix_family(period="5d")
    if df.empty:
        return {}
    latest = df.iloc[-1]
    return {col: float(latest[col]) for col in df.columns if pd.notna(latest[col])}


# ────────────────────────────────────────────────────────────────────────────
# Consolidating the daily 5-day-window snapshots into one clean history
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_VIX_DIR = "data/vix"


def load_vix_history(
    vix_dir: str = DEFAULT_VIX_DIR,
    dropna_vix: bool = False,
) -> pd.DataFrame:
    """Consolidate the per-run VIX snapshots into a single date-indexed frame.

    The daily collector saves a rolling 5-day window (``fetch_vix_family(
    period="5d")``).  On the run date itself some indices have not settled yet
    — typically ``SKEW`` is missing intraday, while ``VIX3M``/``VIX9D`` are only
    populated for the newest row.  Because consecutive snapshots overlap by
    four days, a later file almost always carries the settled value for an
    earlier date, so combining the windows recovers most of those gaps.

    Later files win on conflicts, and non-null values are preferred over nulls.

    Parameters
    ----------
    vix_dir : str
        Directory holding ``vix_YYYY-MM-DD.parquet`` files.
    dropna_vix : bool
        If True, drop dates where the headline ``VIX`` is still missing.
    """
    from pathlib import Path

    files = sorted(Path(vix_dir).glob("vix_*.parquet"))
    if not files:
        logger.warning("No VIX snapshots found in %s", vix_dir)
        return pd.DataFrame()

    combined: Optional[pd.DataFrame] = None
    for path in files:
        try:
            snap = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if snap.empty:
            continue
        snap.index = pd.to_datetime(snap.index)
        # Prefer this (newer) file's non-null cells, keep older ones elsewhere.
        combined = snap if combined is None else snap.combine_first(combined)

    if combined is None or combined.empty:
        return pd.DataFrame()

    combined.index.name = "date"
    combined = combined.sort_index()

    if dropna_vix and "VIX" in combined.columns:
        combined = combined[combined["VIX"].notna()]

    return combined


def save_vix_history(
    vix_dir: str = DEFAULT_VIX_DIR,
    out_path: Optional[str] = None,
) -> "Path":
    """Write the consolidated VIX history to ``{vix_dir}/vix_history.parquet``."""
    from pathlib import Path

    df = load_vix_history(vix_dir)
    path = Path(out_path) if out_path else Path(vix_dir) / "vix_history.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved consolidated VIX history: %d dates -> %s", len(df), path)
    return path


def vix_term_structure_signal(vix_dir: str = DEFAULT_VIX_DIR) -> pd.DataFrame:
    """Return the consolidated history plus common derived vol-regime columns.

    Adds:
      ``vix_ts_slope``  – VIX3M / VIX  (contango > 1, backwardation < 1)
      ``vix_short_ts``  – VIX / VIX9D
      ``vvix_vix``      – VVIX / VIX  (vol-of-vol richness)
    """
    df = load_vix_history(vix_dir)
    if df.empty:
        return df

    out = df.copy()
    if {"VIX3M", "VIX"} <= set(out.columns):
        out["vix_ts_slope"] = out["VIX3M"] / out["VIX"]
    if {"VIX", "VIX9D"} <= set(out.columns):
        out["vix_short_ts"] = out["VIX"] / out["VIX9D"]
    if {"VVIX", "VIX"} <= set(out.columns):
        out["vvix_vix"] = out["VVIX"] / out["VIX"]
    return out


#: Backfill file name.  load_vix_history merges files in name order with later
#: files winning, and this sorts before every dated snapshot, so collected
#: snapshots always take precedence where the two overlap.
BACKFILL_FILENAME = "vix_0000_backfill.parquet"


def backfill_vix_history(
    start: str,
    end: Optional[str] = None,
    vix_dir: str = DEFAULT_VIX_DIR,
) -> pd.DataFrame:
    """Download the VIX family for ``[start, end)`` into the backfill file.

    Unlike option chains, index history is always available, so features for
    historical surfaces (and the SPY-vs-VIX check on them) can use it.
    Re-running merges into the existing backfill (new values win), so a short
    range never erases a longer one already stored; dated snapshots are
    untouched.
    """
    from pathlib import Path

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = fetch_vix_family(start=start, end=end)
    if df.empty:
        logger.error("No VIX-family history returned for %s..%s", start, end)
        return df
    out = Path(vix_dir) / BACKFILL_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = pd.read_parquet(out)
        existing.index = pd.to_datetime(existing.index)
        df = df.combine_first(existing).sort_index()
        df.index.name = "date"
    df.to_parquet(out, engine="pyarrow")
    logger.info("Saved VIX backfill: %d dates (%s -> %s) -> %s",
                len(df), df.index.min().date(), df.index.max().date(), out)
    return df
