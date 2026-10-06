"""Load and normalise the Kaggle SPY implied-volatility dataset.

The Kaggle dataset "SPY Options Implied Volatility" typically ships as a CSV
with columns like:

    date, expiration, strike, call_iv, put_iv, call_bid, call_ask, put_bid,
    put_ask, call_volume, put_volume, call_open_interest, put_open_interest,
    underlying_price  (column names vary by uploader)

This module auto-detects the schema and maps it into the canonical chain
format used by ``src.data.storage``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Known column aliases across popular Kaggle SPY IV datasets                   #
# --------------------------------------------------------------------------- #
_ALIASES: dict[str, list[str]] = {
    "date":              ["date", "quote_date", "data_date", "trade_date", "as_of"],
    "expiration":        ["expiration", "expire_date", "expiry", "expiration_date", "exp_date"],
    "strike":            ["strike", "strike_price"],
    "underlying_price":  ["underlying_price", "stock_price", "spot", "close", "underlying_last",
                          "active_underlying_price"],
    "call_iv":           ["call_iv", "c_iv", "call_implied_volatility", "impl_volatility_c"],
    "put_iv":            ["put_iv", "p_iv", "put_implied_volatility", "impl_volatility_p"],
    "call_bid":          ["call_bid", "c_bid", "best_bid_c", "bid_c"],
    "call_ask":          ["call_ask", "c_ask", "best_offer_c", "ask_c"],
    "put_bid":           ["put_bid", "p_bid", "best_bid_p", "bid_p"],
    "put_ask":           ["put_ask", "p_ask", "best_offer_p", "ask_p"],
    "call_volume":       ["call_volume", "c_volume", "volume_c"],
    "put_volume":        ["put_volume", "p_volume", "volume_p"],
    "call_open_interest": ["call_open_interest", "c_oi", "open_interest_c"],
    "put_open_interest":  ["put_open_interest", "p_oi", "open_interest_p"],
    # Some datasets already have a long-format option_type column
    "option_type":       ["option_type", "cp_flag", "type"],
    "implied_volatility": ["implied_volatility", "iv", "impl_volatility"],
    "bid":               ["bid"],
    "ask":               ["ask"],
    "volume":            ["volume"],
    "open_interest":     ["open_interest", "oi"],
}


def _normalise_name(col: str) -> str:
    """Reduce a raw column name to a comparable key.

    Uploaders wrap headers in brackets (``[C_IV]``), pad them with spaces, or
    use mixed case and separators.  Normalising here means the alias table
    only has to list one spelling per concept.
    """
    key = str(col).strip().lower()
    for ch in "[]()":
        key = key.replace(ch, "")
    return key.strip().replace(" ", "_").replace("-", "_")


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names → actual column names found in *df*."""
    col_lower = {_normalise_name(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in col_lower:
                mapping[canonical] = col_lower[alias]
                break
    return mapping


def _wide_to_long(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Convert wide-format (call_iv, put_iv side by side) to long format."""
    date_col = col_map["date"]
    exp_col = col_map["expiration"]
    strike_col = col_map["strike"]
    spot_col = col_map.get("underlying_price")

    frames: list[pd.DataFrame] = []

    for opt_type in ("call", "put"):
        iv_col = col_map.get(f"{opt_type}_iv")
        bid_col = col_map.get(f"{opt_type}_bid")
        ask_col = col_map.get(f"{opt_type}_ask")
        vol_col = col_map.get(f"{opt_type}_volume")
        oi_col = col_map.get(f"{opt_type}_open_interest")

        if iv_col is None:
            continue

        rec: dict[str, object] = {
            "as_of": df[date_col],
            "expiration": df[exp_col],
            "strike": df[strike_col],
            "option_type": opt_type,
            "implied_volatility_market": df[iv_col],
        }
        if spot_col:
            rec["underlying_price"] = df[spot_col]
        if bid_col:
            rec["bid"] = df[bid_col]
        if ask_col:
            rec["ask"] = df[ask_col]
        if bid_col and ask_col:
            rec["mid"] = (df[bid_col] + df[ask_col]) / 2
        if vol_col:
            rec["volume"] = df[vol_col]
        if oi_col:
            rec["open_interest"] = df[oi_col]

        frames.append(pd.DataFrame(rec))

    return pd.concat(frames, ignore_index=True)


def load_kaggle_spy_iv(filepath: str | Path) -> pd.DataFrame:
    """Load a Kaggle SPY IV file (CSV or Parquet) and return a canonical chain DataFrame.

    The returned DataFrame has the same schema expected by ``storage.save_options_chain``.
    """
    filepath = Path(filepath)
    logger.info("Loading Kaggle data from %s", filepath)

    if filepath.suffix in (".parquet", ".pq"):
        raw = pd.read_parquet(filepath)
    else:
        raw = pd.read_csv(filepath, parse_dates=True)

    logger.info("Raw shape: %s  columns: %s", raw.shape, list(raw.columns))
    col_map = _resolve_columns(raw)
    logger.info("Resolved column mapping: %s", col_map)

    required = {"date", "expiration", "strike"}
    missing = required - set(col_map.keys())
    if missing:
        raise ValueError(
            f"Cannot find required columns {missing} in dataset. "
            f"Available: {list(raw.columns)}"
        )

    # Detect wide vs. long format
    is_wide = "call_iv" in col_map or "put_iv" in col_map
    is_long = "option_type" in col_map and "implied_volatility" in col_map

    if is_wide:
        df = _wide_to_long(raw, col_map)
    elif is_long:
        df = pd.DataFrame({
            "as_of":      raw[col_map["date"]],
            "expiration":  raw[col_map["expiration"]],
            "strike":      raw[col_map["strike"]],
            "option_type": raw[col_map["option_type"]].str.lower().str.strip(),
            "implied_volatility_market": raw[col_map["implied_volatility"]],
        })
        if "underlying_price" in col_map:
            df["underlying_price"] = raw[col_map["underlying_price"]]
        for extra in ("bid", "ask", "volume", "open_interest"):
            if extra in col_map:
                df[extra] = raw[col_map[extra]]
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
    else:
        raise ValueError(
            "Cannot detect dataset format. Need either (call_iv, put_iv) columns "
            "or (option_type, implied_volatility). "
            f"Available: {list(raw.columns)}"
        )

    # Parse dates
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Time to expiry
    df["T"] = (df["expiration"] - df["as_of"]).dt.days / 365.0
    df = df[df["T"] > 0].copy()

    # Ticker
    df["ticker"] = "SPY"

    # Normalise option_type to "call" / "put"
    df["option_type"] = df["option_type"].replace({"C": "call", "P": "put", "c": "call", "p": "put"})

    # Drop rows with missing IV
    df = df.dropna(subset=["implied_volatility_market"])
    df = df[df["implied_volatility_market"] > 0]

    # Also populate implied_volatility for downstream compatibility
    df["implied_volatility"] = df["implied_volatility_market"]

    logger.info("Normalised %d rows from Kaggle dataset", len(df))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Full end-of-day chains in JSON (Alpha Vantage HISTORICAL_OPTIONS format)     #
# --------------------------------------------------------------------------- #
# Kaggle "S&P500 Options (SPY) Implied Volatility (2014-25)" ships one JSON
# file per year, ~1 GB each, of flat contract records with string values
# (contractID, expiration, strike, type, bid, ask, volume, open_interest,
# date, implied_volatility, greeks).  The layout differs by year (a list of
# per-day lists in 2024, an object keyed by date in 2025), and the files are
# too large to json.load at once, so they are streamed record by record.

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def iter_json_days(path: str | Path, chunk_size: int = 1 << 24):
    """Yield the contract records of a chain file, one list per trading day.

    Matches each flat ``{...}`` record (records hold no nested objects, so
    the enclosing lists or date-keyed object are skipped) and groups runs of
    equal ``date``; memory stays at about one day.  Works for either layout.
    """
    buf = ""
    day: list[dict] = []
    current = None
    with open(path, encoding="utf-8") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buf += data
            consumed = 0
            for m in _JSON_OBJECT.finditer(buf):
                rec = json.loads(m.group())
                if day and rec.get("date") != current:
                    yield day
                    day = []
                current = rec.get("date")
                day.append(rec)
                consumed = m.end()
            buf = buf[consumed:]
    if day:
        yield day


def normalise_json_records(records: list[dict], spot: pd.Series,
                           ticker: str = "SPY") -> pd.DataFrame:
    """Records of :func:`iter_json_days` in the canonical chain format.

    The files carry no underlying price, so *spot* (closes indexed by date)
    supplies it; rows on dates without a close are dropped.
    """
    raw = pd.DataFrame.from_records(records)
    if raw.empty:
        return raw
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    df = pd.DataFrame({
        "as_of": pd.to_datetime(raw["date"]),
        "expiration": pd.to_datetime(raw["expiration"]),
        "strike": num("strike"),
        "option_type": raw["type"].str.lower().str.strip(),
        "implied_volatility_market": num("implied_volatility"),
        "bid": num("bid"),
        "ask": num("ask"),
        "volume": num("volume"),
        "open_interest": num("open_interest"),
    })
    df["mid"] = (df["bid"] + df["ask"]) / 2
    spot = spot.copy()
    spot.index = pd.to_datetime(spot.index).normalize()
    df["underlying_price"] = df["as_of"].map(spot)
    df["T"] = (df["expiration"] - df["as_of"]).dt.days / 365.0
    df = df[(df["T"] > 0) & df["underlying_price"].notna()
            & df["option_type"].isin(["call", "put"])].copy()
    df["ticker"] = ticker
    df["implied_volatility"] = df["implied_volatility_market"]
    return df.reset_index(drop=True)
