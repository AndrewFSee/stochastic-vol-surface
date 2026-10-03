"""Data-health checks across the whole collected corpus.

Turns the one-off audit questions — *is the scraper still running, are there
gaps, is any ticker degrading* — into a repeatable report that can be run
daily and diffed over time.

Public API
----------
``coverage_report``   – per-ticker date coverage and gaps
``quality_report``    – per-snapshot row/IV quality statistics
``freshness``         – how stale the newest data is
``full_report``       – all of the above, plus VIX / rates / surface status
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_DIR = "data/options"
DEFAULT_VIX_DIR = "data/vix"
DEFAULT_RATES_DIR = "data/rates"
DEFAULT_SURFACES_DIR = "data/surfaces"

# A snapshot thinner than this is usually a partial scrape, not a quiet market.
MIN_HEALTHY_ROWS = 200
MIN_HEALTHY_EXPIRIES = 4


# ────────────────────────────────────────────────────────────────────────────
# Trading-day helper
# ────────────────────────────────────────────────────────────────────────────


def expected_trading_days(start: date, end: date) -> list[date]:
    """Return NYSE trading days in ``[start, end]``.

    Falls back to weekdays when ``pandas_market_calendars`` is unavailable, in
    which case market holidays will show up as false gaps.
    """
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        days = nyse.valid_days(start.isoformat(), end.isoformat())
        return [pd.Timestamp(d).date() for d in days]
    except ImportError:
        logger.warning(
            "pandas_market_calendars not installed — holidays will be "
            "reported as missing days."
        )
        return [d.date() for d in pd.bdate_range(start, end)]


# ────────────────────────────────────────────────────────────────────────────
# Coverage
# ────────────────────────────────────────────────────────────────────────────


def coverage_report(options_dir: str = DEFAULT_OPTIONS_DIR) -> pd.DataFrame:
    """Return per-ticker coverage: date span, count, and true missing days."""
    from src.surface.batch import available_dates, available_tickers

    tickers = available_tickers(options_dir)
    rows = []
    for t in tickers:
        dts = available_dates(t, options_dir)
        if not dts:
            rows.append({"ticker": t, "n_dates": 0, "first": None, "last": None,
                         "n_missing": 0, "missing": [], "coverage_pct": 0.0})
            continue
        expected = expected_trading_days(min(dts), max(dts))
        missing = sorted(set(expected) - set(dts))
        rows.append({
            "ticker": t,
            "n_dates": len(dts),
            "first": min(dts),
            "last": max(dts),
            "n_expected": len(expected),
            "n_missing": len(missing),
            "missing": [d.isoformat() for d in missing],
            "coverage_pct": 100.0 * len(dts) / max(len(expected), 1),
        })
    return pd.DataFrame(rows)


def freshness(options_dir: str = DEFAULT_OPTIONS_DIR,
              as_of: Optional[date] = None) -> dict:
    """Report how current the newest snapshot is.

    ``trading_days_stale`` counts NYSE sessions strictly after the newest
    stored date up to *as_of*, so a report run before that evening's scrape
    shows 0 rather than a spurious 1.
    """
    from src.surface.batch import available_dates, available_tickers

    as_of = as_of or date.today()
    tickers = available_tickers(options_dir)
    latest_per_ticker = {}
    for t in tickers:
        dts = available_dates(t, options_dir)
        if dts:
            latest_per_ticker[t] = max(dts)

    if not latest_per_ticker:
        return {"latest": None, "trading_days_stale": None, "per_ticker": {}}

    latest = max(latest_per_ticker.values())
    sessions_after = [d for d in expected_trading_days(latest, as_of) if d > latest]

    return {
        "latest": latest,
        "trading_days_stale": len(sessions_after),
        "per_ticker": {k: v.isoformat() for k, v in sorted(latest_per_ticker.items())},
        "lagging_tickers": sorted(k for k, v in latest_per_ticker.items() if v < latest),
    }


# ────────────────────────────────────────────────────────────────────────────
# Quality
# ────────────────────────────────────────────────────────────────────────────


def quality_report(
    options_dir: str = DEFAULT_OPTIONS_DIR,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return one row per stored snapshot with quality statistics."""
    from src.surface.batch import available_dates, available_tickers

    tickers = list(tickers) if tickers else available_tickers(options_dir)
    rows = []
    for t in tickers:
        for d in available_dates(t, options_dir):
            fp = Path(options_dir) / f"ticker={t}" / f"date={d.isoformat()}" / "chain.parquet"
            try:
                df = pd.read_parquet(fp)
            except Exception as exc:
                rows.append({"ticker": t, "as_of": d, "n_rows": 0,
                             "read_error": str(exc)})
                continue

            if df.empty:
                rows.append({"ticker": t, "as_of": d, "n_rows": 0, "read_error": ""})
                continue

            iv = df.get("implied_volatility_market")
            bid, ask = df.get("bid"), df.get("ask")
            rows.append({
                "ticker": t,
                "as_of": d,
                "n_rows": len(df),
                "n_expiries": df["expiration"].nunique() if "expiration" in df else 0,
                "iv_null_frac": float(iv.isna().mean()) if iv is not None else np.nan,
                "iv_nonpos_frac": float((iv <= 0).mean()) if iv is not None else np.nan,
                "iv_median": float(iv.median()) if iv is not None else np.nan,
                "crossed_frac": (float((ask < bid).mean())
                                 if bid is not None and ask is not None else np.nan),
                "dup_rows": int(df.duplicated(
                    [c for c in ["expiration", "strike", "option_type"]
                     if c in df.columns]).sum()),
                "read_error": "",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["thin"] = out["n_rows"] < MIN_HEALTHY_ROWS
    out["few_expiries"] = out["n_expiries"] < MIN_HEALTHY_EXPIRIES
    return out


# ────────────────────────────────────────────────────────────────────────────
# Auxiliary stores
# ────────────────────────────────────────────────────────────────────────────


def vix_status(vix_dir: str = DEFAULT_VIX_DIR) -> dict:
    """Summarise the consolidated VIX history."""
    from src.data.vix_family import load_vix_history

    df = load_vix_history(vix_dir)
    if df.empty:
        return {"n_dates": 0}
    return {
        "n_dates": len(df),
        "first": df.index.min().date().isoformat(),
        "last": df.index.max().date().isoformat(),
        "null_fraction": {c: round(float(df[c].isna().mean()), 4) for c in df.columns},
    }


def rates_status(rates_dir: str = DEFAULT_RATES_DIR) -> dict:
    """Summarise the FRED rate history."""
    from src.data.rates import load_rates_history

    df = load_rates_history(rates_dir)
    if df.empty:
        return {"n_dates": 0}
    return {
        "n_dates": len(df),
        "first": df.index.min().date().isoformat(),
        "last": df.index.max().date().isoformat(),
        "tenors": list(df.columns),
    }


def surface_status(
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    options_dir: str = DEFAULT_OPTIONS_DIR,
) -> pd.DataFrame:
    """Compare built surfaces against available chains, per ticker."""
    from src.surface.batch import available_dates, available_tickers

    rows = []
    for t in available_tickers(options_dir):
        chain_dates = set(available_dates(t, options_dir))
        tdir = Path(surfaces_dir) / f"ticker={t}"
        surf_dates = set()
        if tdir.exists():
            for p in tdir.iterdir():
                if p.is_dir() and (p / "surface.parquet").exists():
                    try:
                        surf_dates.add(date.fromisoformat(p.name.split("=", 1)[1]))
                    except ValueError:
                        pass
        missing = sorted(chain_dates - surf_dates)
        rows.append({
            "ticker": t,
            "n_chains": len(chain_dates),
            "n_surfaces": len(surf_dates),
            "n_unbuilt": len(missing),
            "unbuilt": [d.isoformat() for d in missing[:10]],
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Combined
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class HealthReport:
    """Container for all health sub-reports."""

    coverage: pd.DataFrame
    quality: pd.DataFrame
    surfaces: pd.DataFrame
    freshness: dict
    vix: dict
    rates: dict

    def render(self) -> str:
        """Return a printable multi-section text report."""
        L: list[str] = []
        add = L.append

        add("=" * 72)
        add("DATA HEALTH REPORT")
        add("=" * 72)

        # Freshness
        f = self.freshness
        add("\n-- Freshness --")
        if f.get("latest") is None:
            add("  NO DATA FOUND")
        else:
            stale = f["trading_days_stale"]
            flag = "OK" if stale == 0 else ("WARN" if stale == 1 else "STALE")
            add(f"  latest snapshot : {f['latest']}  ({stale} trading day(s) behind)  [{flag}]")
            if f.get("lagging_tickers"):
                add(f"  lagging tickers : {', '.join(f['lagging_tickers'])}")

        # Coverage
        add("\n-- Coverage --")
        if self.coverage.empty:
            add("  no tickers")
        else:
            for _, r in self.coverage.iterrows():
                add(f"  {r['ticker']:<6} {r['n_dates']:>4} snapshots  "
                    f"{r['first']} -> {r['last']}  "
                    f"{r['coverage_pct']:5.1f}% of sessions  "
                    f"missing={r['n_missing']}")
                if r["n_missing"]:
                    add(f"         gaps: {', '.join(r['missing'][:8])}"
                        + (" ..." if r["n_missing"] > 8 else ""))

        # Quality
        add("\n-- Quality --")
        q = self.quality
        if q.empty:
            add("  no snapshots")
        else:
            add(f"  snapshots: {len(q)}   total rows: {int(q.n_rows.sum()):,}")
            add(f"  thin snapshots (<{MIN_HEALTHY_ROWS} rows): {int(q.thin.sum())}")
            add(f"  snapshots with <{MIN_HEALTHY_EXPIRIES} expiries: {int(q.few_expiries.sum())}")
            add(f"  snapshots with duplicate contracts: {int((q.dup_rows > 0).sum())}")
            add(f"  snapshots with any non-positive IV: {int((q.iv_nonpos_frac > 0).sum())}")
            add(f"  snapshots with crossed quotes:      {int((q.crossed_frac > 0).sum())}")
            add("\n  per-ticker median rows / median IV:")
            g = q.groupby("ticker").agg(
                median_rows=("n_rows", "median"),
                min_rows=("n_rows", "min"),
                median_iv=("iv_median", "median"),
                thin=("thin", "sum"),
            )
            for t, r in g.iterrows():
                add(f"    {t:<6} rows med={r['median_rows']:>7.0f} min={r['min_rows']:>6.0f}"
                    f"   IV med={r['median_iv']:.4f}   thin={int(r['thin'])}")

        # Surfaces
        add("\n-- Surfaces --")
        if self.surfaces.empty:
            add("  none built")
        else:
            for _, r in self.surfaces.iterrows():
                mark = "OK" if r["n_unbuilt"] == 0 else f"{r['n_unbuilt']} unbuilt"
                add(f"  {r['ticker']:<6} {r['n_surfaces']:>4}/{r['n_chains']:<4} built   [{mark}]")

        # Aux stores
        add("\n-- VIX family --")
        if self.vix.get("n_dates"):
            add(f"  {self.vix['n_dates']} dates  {self.vix['first']} -> {self.vix['last']}")
            add(f"  null fraction: {self.vix['null_fraction']}")
        else:
            add("  MISSING")

        add("\n-- Risk-free rates --")
        if self.rates.get("n_dates"):
            add(f"  {self.rates['n_dates']} dates  {self.rates['first']} -> {self.rates['last']}")
            add(f"  tenors: {', '.join(self.rates['tenors'])}")
        else:
            add("  MISSING  (run: python scripts/backfill_rates.py)")

        add("")
        return "\n".join(L)


def full_report(
    options_dir: str = DEFAULT_OPTIONS_DIR,
    vix_dir: str = DEFAULT_VIX_DIR,
    rates_dir: str = DEFAULT_RATES_DIR,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    as_of: Optional[date] = None,
) -> HealthReport:
    """Assemble every sub-report into a single :class:`HealthReport`."""
    return HealthReport(
        coverage=coverage_report(options_dir),
        quality=quality_report(options_dir),
        surfaces=surface_status(surfaces_dir, options_dir),
        freshness=freshness(options_dir, as_of=as_of),
        vix=vix_status(vix_dir),
        rates=rates_status(rates_dir),
    )
