"""Macro and credit series from FRED, joined point-in-time.

Free, full-history series that describe the market regime the volatility
surface sits in:

==================  ============  ===========================================
name                FRED series   meaning
==================  ============  ===========================================
HY_OAS              BAMLH0A0HYM2  high-yield credit spread, % (FRED keeps 3 years)
IG_OAS              BAMLC0A0CM    investment-grade credit spread, % (3 years)
USD_BROAD           DTWEXBGS      broad trade-weighted dollar index
BREAKEVEN_10Y       T10YIE        10-year breakeven inflation, %
REAL_YIELD_10Y      DFII10        10-year TIPS real yield, %
STLFSI              STLFSI4       St. Louis Fed financial stress index (weekly)
NFCI                NFCI          Chicago Fed financial conditions index (weekly)
==================  ============  ===========================================

**Point in time.** FRED dates an observation by the period it describes, not
the day it was published.  The weekly indices appear 5–6 days after their
date, and the daily ones the next day.  :func:`available_asof` shifts each
series to its publication date before it is joined to a trading date, so a
feature dated *t* never uses a number that was not public by *t*.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MACRO_DIR = "data/macro"
HISTORY_FILENAME = "macro_history.parquet"

#: name -> (FRED id, publication lag in calendar days after the observation date)
MACRO_SERIES: dict[str, tuple[str, int]] = {
    "HY_OAS": ("BAMLH0A0HYM2", 1),
    "IG_OAS": ("BAMLC0A0CM", 1),
    "USD_BROAD": ("DTWEXBGS", 1),
    "BREAKEVEN_10Y": ("T10YIE", 1),
    "REAL_YIELD_10Y": ("DFII10", 1),
    "STLFSI": ("STLFSI4", 6),    # week ending Friday, published the next Thursday
    "NFCI": ("NFCI", 5),         # week ending Friday, published the next Wednesday
}


def history_path(macro_dir: str = DEFAULT_MACRO_DIR) -> Path:
    return Path(macro_dir) / HISTORY_FILENAME


def fetch_macro(start: str, end: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """Observations for every series in :data:`MACRO_SERIES`, indexed by FRED date."""
    from fredapi import Fred

    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set")
    fred = Fred(api_key=api_key)
    cols = {}
    for name, (sid, _lag) in MACRO_SERIES.items():
        try:
            cols[name] = fred.get_series(sid, observation_start=start, observation_end=end)
        except Exception as exc:
            logger.warning("FRED %s (%s) failed: %s", name, sid, exc)
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def save_macro_history(df: pd.DataFrame, macro_dir: str = DEFAULT_MACRO_DIR) -> Path:
    """Merge *df* into the stored history (new non-null values win)."""
    out = history_path(macro_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_parquet(out)
        old.index = pd.to_datetime(old.index)
        df = df.combine_first(old)
    df = df.sort_index()
    df.index.name = "date"
    df.to_parquet(out, engine="pyarrow")
    logger.info("Saved macro history: %d dates x %d series -> %s", len(df), df.shape[1], out)
    return out


def load_macro_history(macro_dir: str = DEFAULT_MACRO_DIR) -> pd.DataFrame:
    p = history_path(macro_dir)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def available_asof(history: pd.DataFrame, dates) -> pd.DataFrame:
    """Each series' latest value *published* on or before each of *dates*.

    Returns a frame indexed by *dates* with lower-case ``macro_<name>`` columns.
    """
    target = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    out = pd.DataFrame(index=target)
    for name, (_sid, lag) in MACRO_SERIES.items():
        if name not in history:
            continue
        s = history[name].dropna()
        if s.empty:
            continue
        pub = pd.DataFrame({"published": s.index + pd.Timedelta(days=lag), "value": s.to_numpy()})
        left = pd.DataFrame({"date": target}).reset_index(drop=True)
        m = pd.merge_asof(left.sort_values("date"), pub.sort_values("published"),
                          left_on="date", right_on="published", direction="backward")
        out[f"macro_{name.lower()}"] = m.set_index("date")["value"].reindex(target).to_numpy()
    return out
