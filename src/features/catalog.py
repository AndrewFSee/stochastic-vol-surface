"""The feature table's data contract: every column's meaning, unit and gaps.

Downstream models should read this rather than guess from names.
``describe(col)`` returns one column's entry; ``catalog(columns)`` a table;
``scripts/feature_catalog.py`` writes ``docs/feature_catalog.md`` and
``docs/feature_catalog.json``.  A test checks that every column the table
builder produces has an entry, so the catalog cannot silently fall behind.

Timing (all columns): a row dated *t* uses only information available at
about 16:30 ET on *t*.  Use row *t* to predict anything over ``(t, t+h]``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Entry:
    pattern: str          # regex matched against the full column name
    group: str
    unit: str
    description: str
    nan_when: str = ""


_TENOR = r"(7|14|30|60|91|182|365)d"
_DYN = (r"(atm_30d|atm_91d|vs_30d|rr25_30d|bf25_30d|ts_7_30|ts_30_91|vrp_30d)")

ENTRIES: tuple[Entry, ...] = (
    # Keys and provenance
    Entry(r"ticker", "key", "", "Underlying symbol"),
    Entry(r"date", "key", "date", "Trading date of the 16:30 ET option snapshot"),
    Entry(r"builder", "provenance", "", "Surface-builder version that produced the fits"),
    Entry(r"feature_version", "provenance", "", "Feature-table version; rows of one build share it"),
    # Surface
    Entry(r"spot", "surface", "price", "Underlying price at the snapshot (close)"),
    Entry(rf"atm_{_TENOR}", "ATM term structure", "vol (decimal)",
          "Forward-ATM implied vol at constant maturity (calendar days), interpolated in total variance",
          "tenor outside the listed expiries (often 7d/14d for thin names)"),
    Entry(r"iv_(p10|p25|c25|c10)_(30|91)d", "smile", "vol (decimal)",
          "Implied vol at the 10/25-delta put (p) or call (c), forward Black delta",
          "strike outside the bracketing expiries' quotes"),
    Entry(r"rr(25|10)_(30|91)d", "smile", "vol (decimal)",
          "Risk reversal: call-delta vol minus put-delta vol (negative = puts richer)",
          "either wing missing"),
    Entry(r"bf(25|10)_(30|91)d", "smile", "vol (decimal)",
          "Butterfly: mean of the two delta-wing vols minus ATM vol (smile curvature)",
          "either wing missing"),
    Entry(r"atm_skew_(30|91)d", "smile", "vol per unit log-moneyness",
          "Slope dσ/dk of the smile at the money (finite difference, k = ln(K/F))",
          "tenor not bracketed"),
    Entry(r"atm_curv_(30|91)d", "smile", "vol per unit log-moneyness²",
          "Curvature d²σ/dk² of the smile at the money", "tenor not bracketed"),
    Entry(r"vs_(30|91)d", "variance swap", "vol (decimal)",
          "Model-free variance-swap vol (VIX methodology on the fitted smiles); SPY's vs_30d tracks VIX",
          "tenor not bracketed"),
    Entry(r"ts_(7_30|30_91|30_365)", "term spreads", "vol (decimal)",
          "Longer-tenor minus shorter-tenor ATM vol (positive = upward-sloping)", "either tenor missing"),
    Entry(r"fwd_carry_1y", "carry", "per year",
          "ln(F/S)/T on the parity-forward expiry nearest 1 year: rate − dividend − borrow",
          "no parity forward between 0.5 and 1.5 years"),
    Entry(r"n_expiries", "quality", "count", "Expiries with an accepted SVI fit"),
    Entry(r"nearest_expiry_days", "quality", "days", "Calendar days to the nearest fitted expiry"),
    Entry(r"fit_rmse", "quality", "vol (decimal)", "Median per-expiry SVI fit error"),
    Entry(r"parity_fraction", "quality", "fraction",
          "Share of expiries whose forward came from put-call parity (rest: carry model)"),
    # Realised
    Entry(r"ret_(1|5|21)d", "realised", "log return", "Trailing close-to-close log return (dividend-adjusted)"),
    Entry(r"rv_cc_(10|21|63)d", "realised", "vol (decimal)",
          "Trailing zero-mean close-to-close realised vol, annualised (252)", "fewer prices than the window"),
    Entry(r"rv_yz_21d", "realised", "vol (decimal)", "Trailing 21-day Yang-Zhang realised vol (uses OHLC)"),
    Entry(r"vrp_30d", "premia", "vol (decimal)", "atm_30d minus rv_cc_21d"),
    Entry(r"vrp_var_30d", "premia", "variance", "vs_30d² minus rv_cc_21d² (a variance swap's carry)"),
    # Market and macro
    Entry(r"mkt_(vix|vix3m|vix9d|vix1d|vvix|vxn|vxd|ovx|gvz)", "market", "index points (vol %)",
          "CBOE volatility-index close (VIX family; VXN Nasdaq-100, VXD Dow, OVX oil, GVZ gold); "
          "same for every ticker", "index not published that day; VIX1D starts 2023"),
    Entry(r"mkt_skew", "market", "index points", "CBOE SKEW index close"),
    Entry(r"mkt_move", "market", "index points (bp vol)", "ICE BofA MOVE Treasury-option vol index"),
    Entry(r"macro_(hy|ig)_oas", "macro", "percent",
          "ICE BofA high-yield / investment-grade option-adjusted spread (FRED)",
          "FRED licenses only the last 3 years"),
    Entry(r"macro_usd_broad", "macro", "index", "Nominal broad US dollar index (FRED DTWEXBGS)"),
    Entry(r"macro_breakeven_10y", "macro", "percent", "10-year breakeven inflation (FRED T10YIE)"),
    Entry(r"macro_real_yield_10y", "macro", "percent", "10-year TIPS real yield (FRED DFII10)"),
    Entry(r"macro_(stlfsi|nfci)", "macro", "index (0 = average)",
          "St. Louis Fed / Chicago Fed financial-stress indices (weekly)"),
    # Earnings
    Entry(r"earn_next_date", "earnings", "date",
          "Session the next release moves (after-close releases move the next day); "
          "point-in-time from the first schedule snapshot (Oct 2026), actual dates before",
          "funds/indices; no scheduled release"),
    Entry(r"earn_days_to", "earnings", "trading sessions",
          "Sessions in (t, earn_next_date]: 1 = the release moves tomorrow", "funds/indices"),
    Entry(r"earn_days_since", "earnings", "trading sessions",
          "Sessions since the last release's session (0 on that session)", "funds/indices"),
    Entry(r"earn_implied_move", "earnings", "log return (1 sd)",
          "One-day move implied by the ATM term structure across the next release",
          "release > ~3 months out, missing expiries, or a non-positive estimate"),
    Entry(r"earn_hist_move", "earnings", "log return (1 sd)",
          "RMS excess earnings-day move over the last 8 releases", "fewer than 4 past releases"),
    # Dynamics
    Entry(rf"{_DYN}_d1", "dynamics", "same as the base column",
          "One-session change on the NYSE calendar", "previous session missing (a gap is not bridged)"),
    Entry(rf"{_DYN}_d5", "dynamics", "same as the base column", "Five-session change", "t−5 missing"),
    Entry(rf"{_DYN}_z63", "dynamics", "z-score",
          "(x − trailing 63-session mean) / sd", "fewer than 20 observations"),
    Entry(rf"{_DYN}_pct252", "dynamics", "percentile 0–1",
          "Rank of x within the trailing 252 sessions (all history when shorter)",
          "fewer than 20 observations"),
)

_COMPILED = [(re.compile(rf"^{e.pattern}$"), e) for e in ENTRIES]


def describe(column: str) -> Optional[Entry]:
    """The catalog entry for *column*, or None if it is not documented."""
    for rx, e in _COMPILED:
        if rx.match(column):
            return e
    return None


def catalog(columns: Iterable[str]) -> pd.DataFrame:
    """One row per column: group, unit, description and when it is NaN."""
    rows = []
    for c in columns:
        e = describe(c)
        rows.append({"column": c, **({k: v for k, v in asdict(e).items() if k != "pattern"}
                                      if e else {"group": "UNDOCUMENTED"})})
    return pd.DataFrame(rows)


def undocumented(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if describe(c) is None]
