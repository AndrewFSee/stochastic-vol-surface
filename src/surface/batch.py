"""Batch construction of the historical volatility-surface corpus.

Turns the partitioned raw options store into the standardised surface store
that the neural / calibration / backtest layers consume:

    data/options/ticker={T}/date={D}/chain.parquet     (input,  raw chains)
    data/surfaces/ticker={T}/date={D}/surface.parquet  (output, 25 x 8 grid)

The output layout is the one :mod:`scripts.train` expects.

Design notes
------------
* **Resumable** — existing surfaces are skipped unless ``overwrite=True``, so a
  daily incremental run costs one build per ticker.
* **Real rates** — the discount rate is the FRED curve interpolated to each
  snapshot's ATM tenor, not a hard-coded constant.
* **Quality-controlled** — every surface is scored before it is written; grids
  that are non-finite or wildly out of range are rejected rather than silently
  poisoning downstream training.

Public API
----------
``build_one``        – one (ticker, date) → :class:`SurfaceBuildResult`
``build_corpus``     – sweep the whole store
``surface_path``     – canonical output path
``load_surface_history`` – stacked (n_dates, n_k, n_t) array for training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_DIR = "data/options"
DEFAULT_SURFACES_DIR = "data/surfaces"

# Sanity bounds for a plausible equity/ETF implied-vol grid.  Anything outside
# these is a construction artefact (usually a thin wing extrapolated by SVI),
# not a real quote.
MIN_PLAUSIBLE_IV = 0.01
MAX_PLAUSIBLE_IV = 3.00

# Reference tenor used to pick the discount rate for a snapshot.
RATE_REFERENCE_TENOR = 0.25


# ────────────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class SurfaceBuildResult:
    """Outcome of a single (ticker, as_of) surface build."""

    ticker: str
    as_of: date
    status: str                      # "built" | "skipped" | "rejected" | "failed"
    path: Optional[Path] = None
    n_chain_rows: int = 0
    spot: float = float("nan")
    r: float = float("nan")
    atm_3m: float = float("nan")
    atm_1y: float = float("nan")
    skew_3m: float = float("nan")
    iv_min: float = float("nan")
    iv_max: float = float("nan")
    nan_fraction: float = float("nan")
    message: str = ""

    def as_row(self) -> dict:
        """Flatten to a dict suitable for a report DataFrame."""
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "status": self.status,
            "n_chain_rows": self.n_chain_rows,
            "spot": self.spot,
            "r": self.r,
            "atm_3m": self.atm_3m,
            "atm_1y": self.atm_1y,
            "skew_3m": self.skew_3m,
            "iv_min": self.iv_min,
            "iv_max": self.iv_max,
            "nan_fraction": self.nan_fraction,
            "message": self.message,
        }


@dataclass
class CorpusBuildReport:
    """Aggregate outcome of a corpus sweep."""

    results: list[SurfaceBuildResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return all per-surface results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.as_row() for r in self.results])

    def counts(self) -> dict[str, int]:
        """Return the number of results per status."""
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def summary(self) -> str:
        """Human-readable one-block summary."""
        c = self.counts()
        parts = [f"{k}={v}" for k, v in sorted(c.items())]
        df = self.to_frame()
        line = f"Surface corpus: {len(self.results)} attempted  ({', '.join(parts)})"
        if not df.empty and (df.status == "built").any():
            b = df[df.status == "built"]
            line += (
                f"\n  ATM 3M: median {b.atm_3m.median():.4f}"
                f"  [{b.atm_3m.min():.4f}, {b.atm_3m.max():.4f}]"
                f"\n  IV range across corpus: "
                f"[{b.iv_min.min():.4f}, {b.iv_max.max():.4f}]"
            )
        return line


# ────────────────────────────────────────────────────────────────────────────
# Paths & discovery
# ────────────────────────────────────────────────────────────────────────────


def surface_path(
    ticker: str,
    as_of: date | str,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
) -> Path:
    """Return the canonical output path for one surface."""
    dt = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    return Path(surfaces_dir) / f"ticker={ticker}" / f"date={dt}" / "surface.parquet"


def available_tickers(options_dir: str = DEFAULT_OPTIONS_DIR) -> list[str]:
    """Return tickers present in the raw options store."""
    base = Path(options_dir)
    if not base.exists():
        return []
    return sorted(
        p.name.split("=", 1)[1]
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith("ticker=")
    )


def available_dates(
    ticker: str,
    options_dir: str = DEFAULT_OPTIONS_DIR,
) -> list[date]:
    """Return dates with a stored chain for *ticker*."""
    tdir = Path(options_dir) / f"ticker={ticker}"
    if not tdir.exists():
        return []
    out: list[date] = []
    for p in sorted(tdir.iterdir()):
        if p.is_dir() and p.name.startswith("date="):
            if not (p / "chain.parquet").exists():
                continue
            try:
                out.append(date.fromisoformat(p.name.split("=", 1)[1]))
            except ValueError:
                continue
    return out


# ────────────────────────────────────────────────────────────────────────────
# Quality control
# ────────────────────────────────────────────────────────────────────────────


def score_grid(iv_grid: np.ndarray) -> tuple[bool, str, dict]:
    """Validate a constructed IV grid.

    Returns ``(ok, message, stats)``.  A grid is rejected when it is entirely
    non-finite or when its finite values fall outside plausible vol bounds,
    which indicates a failed fit rather than an unusual market.
    """
    finite = iv_grid[np.isfinite(iv_grid)]
    stats = {
        # Counts +/-inf as well as NaN — both are equally unusable downstream.
        "nan_fraction": float((~np.isfinite(iv_grid)).mean()),
        "iv_min": float(finite.min()) if finite.size else float("nan"),
        "iv_max": float(finite.max()) if finite.size else float("nan"),
    }

    if finite.size == 0:
        return False, "grid is entirely non-finite", stats
    if stats["nan_fraction"] > 0.5:
        return False, f"{stats['nan_fraction']:.0%} of grid is NaN", stats
    if stats["iv_min"] <= MIN_PLAUSIBLE_IV:
        return False, f"implausible min IV {stats['iv_min']:.4f}", stats
    if stats["iv_max"] > MAX_PLAUSIBLE_IV:
        return False, f"implausible max IV {stats['iv_max']:.4f}", stats
    return True, "", stats


# ────────────────────────────────────────────────────────────────────────────
# Single build
# ────────────────────────────────────────────────────────────────────────────


def build_one(
    ticker: str,
    as_of: date,
    *,
    options_dir: str = DEFAULT_OPTIONS_DIR,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    method: str = "svi",
    overwrite: bool = False,
    rates_history: Optional[pd.DataFrame] = None,
    fallback_rate: float = 0.05,
) -> SurfaceBuildResult:
    """Build, quality-check and persist one surface.

    Parameters
    ----------
    ticker, as_of
        Which snapshot to build.
    method
        ``"svi"`` (default) or ``"rbf"``, passed to the interpolator.
    overwrite
        Rebuild even if the output already exists.
    rates_history
        Pre-loaded FRED history; pass it in when looping to avoid re-reading
        the Parquet file for every snapshot.
    fallback_rate
        Used when no curve is available for *as_of*.
    """
    from src.data.storage import load_options_chain
    from src.surface.surface import VolSurface

    out_path = surface_path(ticker, as_of, surfaces_dir)
    if out_path.exists() and not overwrite:
        return SurfaceBuildResult(
            ticker=ticker, as_of=as_of, status="skipped", path=out_path,
            message="already exists",
        )

    dt_str = as_of.isoformat()
    try:
        chain = load_options_chain(ticker, base_dir=options_dir,
                                   start=dt_str, end=dt_str)
    except Exception as exc:
        return SurfaceBuildResult(ticker=ticker, as_of=as_of, status="failed",
                                  message=f"chain load error: {exc}")

    if chain.empty:
        return SurfaceBuildResult(ticker=ticker, as_of=as_of, status="failed",
                                  message="no chain rows")

    # ── Spot ──────────────────────────────────────────────────────────────
    if "underlying_price" not in chain.columns:
        return SurfaceBuildResult(ticker=ticker, as_of=as_of, status="failed",
                                  n_chain_rows=len(chain),
                                  message="chain lacks underlying_price")
    spot_series = chain["underlying_price"].dropna()
    if spot_series.empty:
        return SurfaceBuildResult(ticker=ticker, as_of=as_of, status="failed",
                                  n_chain_rows=len(chain),
                                  message="underlying_price all null")
    # Median guards against the occasional mid-scrape price change.
    spot = float(spot_series.median())

    # ── Discount rate from the real curve ─────────────────────────────────
    try:
        from src.data.rates import get_rate_for_tenor

        r = get_rate_for_tenor(as_of, RATE_REFERENCE_TENOR, history=rates_history)
    except Exception as exc:
        logger.debug("Rate lookup failed for %s %s: %s", ticker, as_of, exc)
        r = fallback_rate

    # ── Build ─────────────────────────────────────────────────────────────
    try:
        vs = VolSurface.from_chain(
            chain, ticker=ticker, as_of=as_of, spot=spot, r=r, method=method,
        )
    except Exception as exc:
        return SurfaceBuildResult(
            ticker=ticker, as_of=as_of, status="failed", n_chain_rows=len(chain),
            spot=spot, r=r, message=f"{type(exc).__name__}: {exc}",
        )

    ok, msg, stats = score_grid(vs.iv_grid)

    result = SurfaceBuildResult(
        ticker=ticker, as_of=as_of, status="built" if ok else "rejected",
        n_chain_rows=len(chain), spot=spot, r=r,
        nan_fraction=stats["nan_fraction"],
        iv_min=stats["iv_min"], iv_max=stats["iv_max"], message=msg,
    )

    if not ok:
        logger.warning("Rejected %s %s: %s", ticker, as_of, msg)
        return result

    # Diagnostics are only meaningful on an accepted grid.
    try:
        result.atm_3m = float(vs.atm_vol(0.25))
        result.atm_1y = float(vs.atm_vol(1.0))
        result.skew_3m = float(vs.skew(0.25))
    except Exception:
        pass

    try:
        vs.save(out_path)
        result.path = out_path
    except Exception as exc:
        result.status = "failed"
        result.message = f"save error: {exc}"

    return result


# ────────────────────────────────────────────────────────────────────────────
# Corpus sweep
# ────────────────────────────────────────────────────────────────────────────


def build_corpus(
    tickers: Optional[Sequence[str]] = None,
    dates: Optional[Iterable[date]] = None,
    *,
    options_dir: str = DEFAULT_OPTIONS_DIR,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    method: str = "svi",
    overwrite: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    progress: bool = True,
) -> CorpusBuildReport:
    """Build surfaces for every (ticker, date) in the raw store.

    Parameters
    ----------
    tickers
        Defaults to every ticker found in *options_dir*.
    dates
        Explicit date list; defaults to each ticker's available dates.
    start, end
        Optional ``YYYY-MM-DD`` bounds applied to the discovered dates.
    overwrite
        Rebuild surfaces that already exist.
    progress
        Log a line per ticker as it completes.
    """
    from src.data.rates import load_rates_history

    if tickers is None:
        tickers = available_tickers(options_dir)
    if not tickers:
        logger.warning("No tickers found in %s", options_dir)
        return CorpusBuildReport()

    # Load the rate history once for the whole sweep.
    try:
        rates_history = load_rates_history()
    except Exception:
        rates_history = pd.DataFrame()
    if rates_history.empty:
        logger.warning(
            "No FRED rate history found — falling back to a constant rate. "
            "Run `python scripts/backfill_rates.py` for accurate forwards."
        )

    report = CorpusBuildReport()

    for tkr in tickers:
        tkr_dates = list(dates) if dates is not None else available_dates(tkr, options_dir)
        if start:
            tkr_dates = [d for d in tkr_dates if d.isoformat() >= start]
        if end:
            tkr_dates = [d for d in tkr_dates if d.isoformat() <= end]

        for d in tkr_dates:
            res = build_one(
                tkr, d,
                options_dir=options_dir, surfaces_dir=surfaces_dir,
                method=method, overwrite=overwrite, rates_history=rates_history,
            )
            report.results.append(res)

        if progress:
            done = [r for r in report.results if r.ticker == tkr]
            built = sum(1 for r in done if r.status == "built")
            skipped = sum(1 for r in done if r.status == "skipped")
            bad = sum(1 for r in done if r.status in ("failed", "rejected"))
            logger.info("%-6s %3d dates: %3d built, %3d skipped, %3d failed/rejected",
                        tkr, len(done), built, skipped, bad)

    return report


# ────────────────────────────────────────────────────────────────────────────
# Reading the corpus back
# ────────────────────────────────────────────────────────────────────────────


def load_surfaces(
    ticker: str,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Load a ticker's surfaces as ``{date: VolSurface}``.

    This is the shape :func:`src.backtest.engine.run_backtest` consumes.
    Surfaces that fail to load are skipped with a warning rather than aborting
    a long backtest.
    """
    from src.surface.surface import VolSurface

    tdir = Path(surfaces_dir) / f"ticker={ticker}"
    if not tdir.exists():
        logger.warning("No surfaces for %s at %s", ticker, tdir)
        return {}

    out: dict = {}
    for ddir in sorted(tdir.iterdir()):
        if not ddir.is_dir() or not ddir.name.startswith("date="):
            continue
        dt_str = ddir.name.split("=", 1)[1]
        if start and dt_str < start:
            continue
        if end and dt_str > end:
            continue
        fp = ddir / "surface.parquet"
        if not fp.exists():
            continue
        try:
            out[date.fromisoformat(dt_str)] = VolSurface.load(fp)
        except Exception as exc:
            logger.warning("Could not load surface %s: %s", fp, exc)

    return out


def load_surface_history(
    ticker: str,
    surfaces_dir: str = DEFAULT_SURFACES_DIR,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[list[date], np.ndarray, np.ndarray, np.ndarray]:
    """Load a ticker's surfaces as a stacked array for model training.

    Returns
    -------
    dates   : list[date]                     length n
    k_grid  : np.ndarray  shape (n_k,)
    t_grid  : np.ndarray  shape (n_t,)
    grids   : np.ndarray  shape (n, n_k, n_t)

    Snapshots whose grid shape differs from the first one are skipped, so the
    returned array is always rectangular.
    """
    tdir = Path(surfaces_dir) / f"ticker={ticker}"
    if not tdir.exists():
        return [], np.array([]), np.array([]), np.empty((0, 0, 0))

    dates_out: list[date] = []
    grids: list[np.ndarray] = []
    k_grid = t_grid = None

    for ddir in sorted(tdir.iterdir()):
        if not ddir.is_dir() or not ddir.name.startswith("date="):
            continue
        dt_str = ddir.name.split("=", 1)[1]
        if start and dt_str < start:
            continue
        if end and dt_str > end:
            continue
        fp = ddir / "surface.parquet"
        if not fp.exists():
            continue

        df = pd.read_parquet(fp)
        if "implied_vol" not in df.columns or df.empty:
            continue

        pivot = df.pivot_table(
            index="log_moneyness", columns="tenor", values="implied_vol",
            aggfunc="mean",
        ).sort_index()
        pivot = pivot[sorted(pivot.columns)]

        if k_grid is None:
            k_grid = pivot.index.to_numpy()
            t_grid = pivot.columns.to_numpy()
        elif pivot.shape != (len(k_grid), len(t_grid)):
            logger.warning("Shape mismatch for %s %s — skipping", ticker, dt_str)
            continue

        grids.append(pivot.to_numpy())
        dates_out.append(date.fromisoformat(dt_str))

    if not grids:
        return [], np.array([]), np.array([]), np.empty((0, 0, 0))

    return dates_out, k_grid, t_grid, np.stack(grids)
