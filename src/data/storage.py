"""Partitioned Parquet storage with deduplication and integrity checks.

Directory layout
----------------
{base_dir}/ticker={TICKER}/date={YYYY-MM-DD}/chain.parquet

Public API
----------
save_options_chain(df, base_dir) -> list[Path]
load_options_chain(ticker, base_dir, start, end) -> pd.DataFrame
list_available_dates(ticker, base_dir) -> list[date]
check_integrity(ticker, base_dir) -> StorageIntegrityReport
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = "data/options"


def save_options_chain(
    df: pd.DataFrame,
    base_dir: str = DEFAULT_BASE_DIR,
    *,
    validate: bool = True,
) -> list[Path]:
    """Write *df* to Hive-style partitioned Parquet (ticker / date).

    Deduplicates rows if data already exists for that partition.
    Optionally validates against :class:`RawOptionsChainSchema` before writing.
    Returns list of written partition paths.
    """
    if df.empty:
        return []

    # Schema validation
    if validate:
        try:
            from src.data.schema import validate_raw_chain
            df = validate_raw_chain(df, lazy=True)
        except Exception as exc:
            logger.warning("Schema validation issues (saving anyway): %s", exc)

    base = Path(base_dir)
    written: list[Path] = []

    # Ensure partition keys exist
    if "ticker" not in df.columns:
        df = df.assign(ticker="UNKNOWN")
    if "as_of" not in df.columns:
        df = df.assign(as_of=pd.Timestamp(date.today()))

    for (tkr, dt), group in df.groupby(["ticker", "as_of"]):
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        part_dir = base / f"ticker={tkr}" / f"date={dt_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "chain.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            merged = pd.concat([existing, group], ignore_index=True)
            # Deduplicate on natural key
            dedup_cols = [c for c in ["strike", "expiration", "option_type"] if c in merged.columns]
            if dedup_cols:
                merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
            merged.to_parquet(out_path, index=False, engine="pyarrow")
            logger.info("Updated %s (%d rows)", out_path, len(merged))
        else:
            group.to_parquet(out_path, index=False, engine="pyarrow")
            logger.info("Created %s (%d rows)", out_path, len(group))

        written.append(out_path)

    return written


def load_options_chain(
    ticker: str = "SPY",
    base_dir: str = DEFAULT_BASE_DIR,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load option-chain data for *ticker* across available dates.

    Parameters
    ----------
    ticker : str
    base_dir : str
    start, end : str "YYYY-MM-DD" or None
        Date-range filter (inclusive).

    Returns
    -------
    pd.DataFrame  sorted by (as_of, strike, option_type).
    """
    ticker_dir = Path(base_dir) / f"ticker={ticker}"
    if not ticker_dir.exists():
        logger.warning("No data directory for %s at %s", ticker, ticker_dir)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for date_dir in sorted(ticker_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("date="):
            continue

        dt_str = date_dir.name.split("=", 1)[1]

        # Date range filter
        if start and dt_str < start:
            continue
        if end and dt_str > end:
            continue

        parquet_file = date_dir / "chain.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            # Ensure as_of column
            if "as_of" not in df.columns:
                df["as_of"] = pd.Timestamp(dt_str)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["as_of", "strike", "option_type"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    return result


def list_available_dates(
    ticker: str = "SPY",
    base_dir: str = DEFAULT_BASE_DIR,
) -> list[date]:
    """Return sorted list of dates that have data for *ticker*."""
    ticker_dir = Path(base_dir) / f"ticker={ticker}"
    if not ticker_dir.exists():
        return []

    dates: list[date] = []
    for date_dir in sorted(ticker_dir.iterdir()):
        if date_dir.is_dir() and date_dir.name.startswith("date="):
            dt_str = date_dir.name.split("=", 1)[1]
            try:
                dates.append(date.fromisoformat(dt_str))
            except ValueError:
                pass
    return dates


def check_integrity(
    ticker: str = "SPY",
    base_dir: str = DEFAULT_BASE_DIR,
) -> dict:
    """Run integrity checks on the stored data for *ticker*.

    Returns a dict compatible with :class:`StorageIntegrityReport`:
    date range, row count, missing weekdays, duplicates, schema errors.
    """
    available = list_available_dates(ticker, base_dir)
    if not available:
        return {
            "ticker": ticker,
            "date_range": ("", ""),
            "n_dates": 0,
            "n_rows": 0,
            "missing_dates": [],
            "duplicate_dates": [],
            "schema_errors": [],
        }

    # Check for missing weekdays
    first, last = min(available), max(available)
    all_weekdays: list[date] = []
    d = first
    while d <= last:
        if d.weekday() < 5:  # Mon-Fri
            all_weekdays.append(d)
        d += timedelta(days=1)
    available_set = set(available)
    missing = [d.isoformat() for d in all_weekdays if d not in available_set]

    # Check for duplicate dates
    from collections import Counter
    date_counts = Counter(available)
    duplicates = [d.isoformat() for d, n in date_counts.items() if n > 1]

    # Count total rows
    total_rows = 0
    schema_errors: list[str] = []
    for dt in available:
        parquet_file = (
            Path(base_dir) / f"ticker={ticker}" / f"date={dt.isoformat()}" / "chain.parquet"
        )
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                total_rows += len(df)
            except Exception as exc:
                schema_errors.append(f"{dt}: read error: {exc}")

    return {
        "ticker": ticker,
        "date_range": (first.isoformat(), last.isoformat()),
        "n_dates": len(available),
        "n_rows": total_rows,
        "missing_dates": missing[:50],  # cap for readability
        "duplicate_dates": duplicates,
        "schema_errors": schema_errors,
    }
