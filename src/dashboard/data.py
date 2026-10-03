"""Data shaping for the dashboard — pure functions, no Streamlit.

Two sources:

* the **feature table** (overview, history, quality), and
* the **per-expiry fits** stored with each surface (smiles, term structure,
  delta × tenor grids), queried through
  :class:`~src.surface.slices.ExpirySurface`.

Market quotes for the smile view are rebuilt from the raw chain using each
expiry's stored discount rate, so they come out on exactly the forward the
fit used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.surface_features import TENORS_DAYS

#: Columns of the delta × tenor grid, low strike → high strike.
DELTA_COLUMNS: dict[str, float] = {
    "10Δ put": -0.10, "25Δ put": -0.25, "ATM": 0.0, "25Δ call": 0.25, "10Δ call": 0.10,
}

#: The overview's trend sparkline length (trading days).
SPARK_DAYS = 63


# ────────────────────────────────────────────────────────────────────────────
# Feature table
# ────────────────────────────────────────────────────────────────────────────


def dates_for(features: pd.DataFrame, ticker: str) -> list[pd.Timestamp]:
    """Dates with a feature row for *ticker*, newest first."""
    d = features.loc[features["ticker"] == ticker, "date"]
    return sorted(pd.to_datetime(d).unique(), reverse=True)


def on_or_before(dates, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Latest date in *dates* that is not after *target*."""
    eligible = [d for d in dates if d <= target]
    return max(eligible) if eligible else None


def overview(features: pd.DataFrame, as_of: pd.Timestamp, stale_days: int = 5) -> pd.DataFrame:
    """One row per ticker: its latest features on or before *as_of*.

    Tickers whose latest row is more than *stale_days* calendar days old are
    dropped rather than shown as if current.  ``atm_30d_trend`` holds the last
    :data:`SPARK_DAYS` values of 30d ATM vol, for a sparkline.
    """
    rows = []
    for tkr, g in features[features["date"] <= as_of].groupby("ticker"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        if (as_of - last["date"]).days > stale_days:
            continue
        row = last.to_dict()
        row["atm_30d_trend"] = g["atm_30d"].tail(SPARK_DAYS).dropna().tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def history(features: pd.DataFrame, ticker: str, start: Optional[pd.Timestamp] = None,
            end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """A ticker's feature rows in ``[start, end]``, oldest first."""
    g = features[features["ticker"] == ticker]
    if start is not None:
        g = g[g["date"] >= start]
    if end is not None:
        g = g[g["date"] <= end]
    return g.sort_values("date").reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Surfaces
# ────────────────────────────────────────────────────────────────────────────


def load_vol_surface(ticker: str, as_of, surfaces_dir: str = "data/surfaces"):
    """The stored surface for (*ticker*, *as_of*), or None if absent/legacy."""
    from src.surface.batch import surface_path
    from src.surface.surface import VolSurface

    path = surface_path(ticker, pd.Timestamp(as_of).date(), surfaces_dir)
    if not path.exists():
        return None
    vs = VolSurface.load(path)
    return vs if vs.slices else None


def expiry_table(vs) -> pd.DataFrame:
    """The fitted expiries: date, days, forward, quote count and fit error."""
    rows = [{
        "expiration": s.get("expiration"),
        "days": round(s["T"] * 365),
        "T": s["T"],
        "forward": s["forward"],
        "forward_source": s["forward_source"],
        "quotes": s["n_quotes"],
        "fit_rmse_pts": 100 * s["rmse_iv"],
    } for s in vs.slices]
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def default_expiries(vs, targets_days=(30, 91)) -> list[str]:
    """The listed expiries closest to *targets_days*."""
    tbl = expiry_table(vs)
    picks = []
    for t in targets_days:
        e = tbl.iloc[(tbl["days"] - t).abs().argmin()]["expiration"]
        if e not in picks:
            picks.append(e)
    return picks


def _slice_for(vs, expiration: str) -> dict:
    return next(s for s in vs.slices if s.get("expiration") == expiration)


def smile_curves(vs, expirations: list[str], n: int = 121, pad: float = 0.02) -> pd.DataFrame:
    """Fitted smiles over each expiry's quoted strike range (plus a margin).

    Columns: ``expiration, days, k, moneyness`` (strike / forward, %) and
    ``iv``.  The curve stops near the last quote; beyond it is extrapolation.
    """
    from src.surface.slices import SliceFit

    frames = []
    for e in expirations:
        s = SliceFit(**_slice_for(vs, e))
        k = np.linspace(s.k_min - pad, s.k_max + pad, n)
        frames.append(pd.DataFrame({
            "expiration": e, "days": round(s.T * 365), "k": k,
            "moneyness": 100 * np.exp(k),
            "iv": np.sqrt(np.maximum(s.total_variance_extrapolated(k), 0) / s.T),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def smile_quotes(vs, ticker: str, as_of, expirations: list[str],
                 options_dir: str = "data/options") -> pd.DataFrame:
    """Market OTM quotes behind the fitted smiles.

    Columns: ``expiration, strike, is_call, k, moneyness, iv, half_spread``
    (half the bid-ask in vol terms — the quote's uncertainty band).
    """
    from src.data.storage import load_options_chain
    from src.surface.slices import otm_quotes

    dt = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    chain = load_options_chain(ticker, base_dir=options_dir, start=dt, end=dt)
    if chain.empty:
        return pd.DataFrame()

    # Discount each expiry at the rate the fit used, so parity reproduces the
    # stored forward exactly.
    rates = {round(s["T"], 8): -math.log(s["discount"]) / s["T"] for s in vs.slices}
    q, _ = otm_quotes(chain, vs.spot, lambda T: rates.get(round(T, 8), vs.r))
    if q.empty or "expiration" not in q.columns:
        return pd.DataFrame()
    q["expiration"] = pd.to_datetime(q["expiration"]).dt.strftime("%Y-%m-%d")
    q = q[q["expiration"].isin(expirations)].copy()
    q["moneyness"] = 100 * np.exp(q["k"])
    q["half_spread"] = 0.5 * q["spread_iv"]
    return q[["expiration", "strike", "is_call", "k", "moneyness", "iv", "half_spread"]]


def delta_markers(vs, expiration: str) -> dict[str, float]:
    """Moneyness (%) of the 25-delta put and call at an expiry."""
    from src.surface.slices import ExpirySurface

    s = _slice_for(vs, expiration)
    surf = ExpirySurface(vs.slices)
    return {name: 100 * math.exp(surf.delta_strike(d, s["T"]))
            for name, d in (("25Δ put", -0.25), ("25Δ call", 0.25))}


def term_structure(vs, n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ATM vol across tenors: a dense curve and the listed-expiry points.

    The curve spans the listed expiries only; nothing is extrapolated.
    """
    from src.surface.slices import ExpirySurface

    surf = ExpirySurface(vs.slices)
    # Log-spaced (the short end has weekly expiries a few days apart) and
    # including every listed expiry, so the curve passes through its points.
    T = np.union1d(np.geomspace(surf.t_min, surf.t_max, n), surf.Ts)
    curve = pd.DataFrame({"days": T * 365, "iv": [float(surf.iv(0.0, t)) for t in T]})
    points = pd.DataFrame({
        "days": surf.Ts * 365,
        "iv": [float(surf.iv(0.0, t)) for t in surf.Ts],
        "expiration": [s.expiration for s in surf.slices],
    })
    return curve, points


def delta_tenor_grid(vs, allow_extrapolation: bool = False) -> pd.DataFrame:
    """Implied vol on a tenor × delta grid (rows tenors, columns deltas).

    Cells outside the listed expiries or the quoted strikes are NaN.
    """
    from src.surface.slices import ExpirySurface

    surf = ExpirySurface(vs.slices)
    out = pd.DataFrame(index=list(TENORS_DAYS), columns=list(DELTA_COLUMNS), dtype=float)
    for name, days in TENORS_DAYS.items():
        T = days / 365
        for col, delta in DELTA_COLUMNS.items():
            k = 0.0 if delta == 0 else surf.delta_strike(delta, T)
            ok = allow_extrapolation or (surf.brackets(T) and bool(surf.is_observed(k, T)))
            out.loc[name, col] = float(surf.iv(k, T)) if (ok and np.isfinite(k)) else np.nan
    return out


# ────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────────────


def pts(x: float) -> str:
    """A vol difference in vol points, signed: ``-2.8 pts``."""
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:+.1f} pts"


def pct(x: float) -> str:
    """A vol level in percent: ``12.7%``."""
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def surfaces_available(surfaces_dir: str) -> bool:
    return Path(surfaces_dir).exists() and any(Path(surfaces_dir).glob("ticker=*"))
