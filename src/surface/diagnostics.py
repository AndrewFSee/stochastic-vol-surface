"""Accuracy diagnostics for a built surface corpus.

The synthetic test suite checks the code; these check the *numbers* against
an independent benchmark.  For SPY that benchmark is VIX: a 30-day variance
swap on the S&P 500, computed by CBOE from SPX options.  Rebuilding the same
quantity from our fitted SPY smiles should track it closely — any surface
noise shows up directly as disagreement.

Public API
----------
``variance_swap_vol``   – model-free vol of one surface at a target tenor
``surface_series``      – per-date ATM / variance-swap / fit-quality series
``noise_stats``         – daily-change statistics for any series
``benchmark_against``   – agreement statistics between two series
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def variance_swap_vol(slices: list[dict], tenor: float = 30 / 365) -> float:
    """Model-free (VIX-style) vol at *tenor* from a surface's fitted slices.

    See :meth:`src.surface.slices.ExpirySurface.variance_swap_vol`.
    """
    from src.surface.slices import ExpirySurface

    if len(slices) < 2:
        return float("nan")
    return ExpirySurface(slices).variance_swap_vol(tenor)


def surface_series(
    ticker: str,
    surfaces_dir: str = "data/surfaces",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Per-date summary of a ticker's corpus, for benchmarking and QC.

    Columns: ``atm_1m``, ``atm_3m`` (from the grid), ``vs_30d`` (variance
    swap, when slice fits are stored), ``fit_rmse``, ``observed_fraction``,
    ``builder``.  Vols are in percent.
    """
    from src.surface.batch import load_surfaces

    rows = []
    for d, vs in load_surfaces(ticker, surfaces_dir=surfaces_dir,
                               start=start, end=end).items():
        rows.append({
            "date": pd.Timestamp(d),
            "atm_1m": 100 * vs.atm_vol(1 / 12),
            "atm_3m": 100 * vs.atm_vol(0.25),
            "vs_30d": 100 * variance_swap_vol(vs.slices) if vs.slices else np.nan,
            "fit_rmse": (100 * float(np.median([s["rmse_iv"] for s in vs.slices]))
                         if vs.slices else np.nan),
            "observed_fraction": (float(vs.observed.mean())
                                  if vs.observed is not None else np.nan),
            "builder": vs.builder,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def noise_stats(series: pd.Series) -> dict[str, float]:
    """Daily-change statistics of one series.

    A strongly *negative* lag-1 autocorrelation of daily changes is the
    signature of measurement noise: a spurious jump one day reverts the next.
    Real implied-vol changes are close to uncorrelated day to day.
    """
    d = series.dropna().diff().dropna()
    return {
        "n": int(len(d)),
        "sd_change": float(d.std()),
        "lag1_autocorr": float(d.autocorr()) if len(d) > 2 else float("nan"),
        "max_abs_change": float(d.abs().max()) if len(d) else float("nan"),
    }


def benchmark_against(series: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Agreement between *series* and an external *benchmark* on common dates."""
    j = pd.concat([series.rename("x"), benchmark.rename("b")], axis=1,
                  join="inner").dropna()
    if len(j) < 3:
        return {"n": len(j)}
    dj = j.diff().dropna()
    return {
        "n": int(len(j)),
        "level_corr": float(j["x"].corr(j["b"])),
        "change_corr": float(dj["x"].corr(dj["b"])),
        "mean_diff": float((j["x"] - j["b"]).mean()),
        "mean_abs_diff": float((j["x"] - j["b"]).abs().mean()),
        "sd_change_ratio": float(dj["x"].std() / dj["b"].std()),
    }
