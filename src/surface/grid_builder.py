"""Convert a raw options chain DataFrame into a standardised log-moneyness × tenor grid.

Pipeline
--------
1. Clean & filter the raw chain (expiry, OI, spread)
2. Compute forward price per expiry  (F = S · e^{rT})
3. Compute IV from mid-prices  (fast Newton + Brent fallback)
4. Map to log-forward-moneyness  k = ln(K / F)
5. Filter moneyness range
6. Interpolate to standardised grid via SVI-per-slice + cubic spline across tenors
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.surface.implied_vol import implied_vol_vectorized

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Grid configuration helpers
# ────────────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "surface_grid.yaml"

#: If at least this fraction of rows carry a positive mid, treat the price
#: column as trustworthy and filter on it as usual.
MIN_PRICED_FRACTION = 0.10


def load_grid_config(path: Path = _CONFIG_PATH) -> dict:
    """Load the surface grid configuration from *path*."""
    with open(path) as f:
        return yaml.safe_load(f)


def default_k_grid(cfg: Optional[dict] = None) -> np.ndarray:
    """Return the default log-moneyness knot vector from config."""
    if cfg is None:
        cfg = load_grid_config()
    g = cfg["grid"]["log_moneyness"]
    return np.linspace(g["min"], g["max"], g["n_points"])


def default_t_grid(cfg: Optional[dict] = None) -> np.ndarray:
    """Return the default tenor knot vector from config."""
    if cfg is None:
        cfg = load_grid_config()
    return np.array(cfg["grid"]["tenor"]["values"], dtype=float)


# ────────────────────────────────────────────────────────────────────────────
# Steps 1-5:  raw chain  →  scatter DataFrame
# ────────────────────────────────────────────────────────────────────────────

def build_surface_grid(
    chain: pd.DataFrame,
    spot: float,
    r: float = 0.05,
    moneyness_range: tuple[float, float] = (-0.40, 0.20),
    n_moneyness: int = 25,
    min_T: float = 1 / 365,
    min_open_interest: int = 0,
    use_forward: bool = True,
) -> pd.DataFrame:
    """Convert a raw options chain to a scatter DataFrame.

    Returns columns: log_moneyness, T, implied_volatility, option_type,
    strike, forward.

    Parameters
    ----------
    chain : pd.DataFrame
        Must have columns: strike, T, option_type, and one of
        mid / bid+ask / last_price.
        Optional: open_interest, implied_volatility_market.
    spot : float
        Current underlying spot price.
    r : float
        Risk-free rate (annualised, continuously compounded).
    moneyness_range : tuple
        (min, max) log-moneyness filter.
    n_moneyness : int
        Informational (not consumed here — used downstream).
    min_T : float
        Minimum time-to-expiry (years) to include.
    min_open_interest : int
        Minimum open interest filter.
    use_forward : bool
        If True, k = ln(K / F) where F = S·exp(rT).
        If False, k = ln(K / S) (legacy behaviour).
    """
    df = chain.copy()

    # ── Basic validation ──────────────────────────────────────────────────
    if "T" not in df.columns:
        raise ValueError("chain must have column 'T' (time-to-expiry in years)")
    df = df[df["T"] > min_T].copy()

    if "open_interest" in df.columns and min_open_interest > 0:
        df = df[df["open_interest"].fillna(0) >= min_open_interest]

    # ── Do we already have a usable quoted IV? ────────────────────────────
    iv_col = next(
        (c for c in ("implied_volatility_market", "implied_volatility")
         if c in df.columns and df[c].gt(0).any()),
        None,
    )

    # ── Compute mid price ─────────────────────────────────────────────────
    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        elif "last_price" in df.columns:
            df["mid"] = df["last_price"]
        elif iv_col is None:
            raise ValueError("chain must have 'mid', 'bid'/'ask', or 'last_price'")

    df = df.dropna(subset=["strike", "T", "option_type"])

    # Normally a non-positive mid marks an untradeable quote and the row is
    # dropped.  But some historical sources publish a settled IV with a blank
    # bid-ask for a whole session; there the price column is uninformative
    # rather than the row being bad, so dropping on it would discard the
    # entire day.  Only relax the filter when prices are globally unusable
    # *and* a quoted IV is available to stand in for them.
    if "mid" in df.columns:
        usable_price = df["mid"].gt(0)
        if iv_col is None or usable_price.mean() >= MIN_PRICED_FRACTION:
            df = df.dropna(subset=["mid"])
            df = df[df["mid"] > 0]
        else:
            logger.info(
                "Only %.1f%% of rows have a positive mid — falling back to the "
                "quoted IV column and skipping the price filter",
                100 * usable_price.mean(),
            )

    # ── Forward price per row ─────────────────────────────────────────────
    if use_forward:
        df["forward"] = spot * np.exp(r * df["T"].to_numpy())
    else:
        df["forward"] = spot

    # ── Compute IV ────────────────────────────────────────────────────────
    if iv_col is not None:
        df["implied_volatility"] = df[iv_col]
        df = df.dropna(subset=["implied_volatility"])
        df = df[df["implied_volatility"] > 0]
    else:
        ivs = implied_vol_vectorized(
            prices=df["mid"].to_numpy(),
            S=spot,
            K=df["strike"].to_numpy(),
            T=df["T"].to_numpy(),
            r=r,
            option_types=df["option_type"].tolist(),
        )
        df = df.assign(implied_volatility=ivs)
        df = df.dropna(subset=["implied_volatility"])
        df = df[df["implied_volatility"] > 0]

    # ── Log-moneyness  k = ln(K / F) ─────────────────────────────────────
    df["log_moneyness"] = np.log(df["strike"].to_numpy() / df["forward"].to_numpy())

    # ── Filter moneyness range ────────────────────────────────────────────
    df = df[
        (df["log_moneyness"] >= moneyness_range[0])
        & (df["log_moneyness"] <= moneyness_range[1])
    ]

    out_cols = [
        "log_moneyness", "T", "implied_volatility",
        "option_type", "strike", "forward",
    ]
    return df[[c for c in out_cols if c in df.columns]].reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Step 6:  scatter  →  regular grid  (SVI-per-slice + spline, or RBF)
# ────────────────────────────────────────────────────────────────────────────

def interpolate_to_grid(
    surface_df: pd.DataFrame,
    k_grid: Optional[np.ndarray] = None,
    tenor_grid: Optional[np.ndarray] = None,
    method: str = "svi",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a scatter surface onto a regular (k, T) grid.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of :func:`build_surface_grid`.
    k_grid, tenor_grid : array-like or None
        Defaults to config values in ``config/surface_grid.yaml``.
    method : str
        ``"svi"``  – SVI fit per tenor slice, cubic spline across tenors.
        ``"rbf"``  – thin-plate-spline RBF (fast fall-back).

    Returns
    -------
    k_nodes  : np.ndarray  shape (n_k,)
    t_nodes  : np.ndarray  shape (n_t,)
    iv_grid  : np.ndarray  shape (n_k, n_t)  — NaN where unreliable
    """
    if k_grid is None:
        k_grid = default_k_grid()
    if tenor_grid is None:
        tenor_grid = default_t_grid()

    if method == "svi":
        return _interpolate_svi_spline(surface_df, k_grid, tenor_grid)
    return _interpolate_rbf(surface_df, k_grid, tenor_grid)


# ── SVI → cubic-spline pipeline ──────────────────────────────────────────

def _interpolate_svi_spline(
    df: pd.DataFrame,
    k_grid: np.ndarray,
    tenor_grid: np.ndarray,
    min_points_per_slice: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SVI per tenor bin, then cubic-spline across tenors."""
    from scipy.interpolate import CubicSpline

    from src.surface.interpolation import interpolate_svi_slice, svi_raw

    n_k = len(k_grid)
    n_t = len(tenor_grid)
    tv_slices: dict[float, np.ndarray] = {}
    fitted_tenors: list[float] = []

    for T_target in tenor_grid:
        T_lo, T_hi = T_target * 0.70, T_target * 1.30
        sdf = df[(df["T"] >= T_lo) & (df["T"] <= T_hi)]

        if len(sdf) < min_points_per_slice:
            continue

        k_raw = sdf["log_moneyness"].to_numpy()
        iv_raw = sdf["implied_volatility"].to_numpy()
        T_actual = float(sdf["T"].median())
        w_raw = iv_raw ** 2 * T_actual  # total variance

        try:
            _w_fit, params = interpolate_svi_slice(k_raw, w_raw)
            w_grid = svi_raw(k_grid, **params)
            w_grid = np.maximum(w_grid, 1e-8)
            tv_slices[T_target] = w_grid
            fitted_tenors.append(T_target)
        except Exception as exc:
            logger.warning("SVI fit failed for T=%.3f: %s", T_target, exc)

    if len(fitted_tenors) < 2:
        logger.warning(
            "Only %d tenor slices fitted — falling back to RBF",
            len(fitted_tenors),
        )
        return _interpolate_rbf(df, k_grid, tenor_grid)

    fitted_arr = np.array(fitted_tenors)
    tv_matrix = np.array([tv_slices[t] for t in fitted_tenors])  # (n_fitted, n_k)

    iv_grid = np.full((n_k, n_t), np.nan)
    for j in range(n_k):
        col = tv_matrix[:, j]
        if len(col) >= 2:
            cs = CubicSpline(fitted_arr, col, extrapolate=True)
            for ti, T in enumerate(tenor_grid):
                w = max(float(cs(T)), 1e-8)
                iv_grid[j, ti] = math.sqrt(w / max(T, 1e-6))

    return k_grid, tenor_grid, iv_grid


# ── RBF fall-back ────────────────────────────────────────────────────────

#: RBFInterpolator needs more points than it has dimensions; below this it
#: raises deep inside SciPy with an opaque reshape error.
_MIN_RBF_POINTS = 4


def _interpolate_rbf(
    df: pd.DataFrame,
    k_grid: np.ndarray,
    tenor_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin-plate-spline RBF interpolation (fast fall-back)."""
    from scipy.interpolate import RBFInterpolator

    if len(df) < _MIN_RBF_POINTS:
        raise ValueError(
            f"cannot interpolate a surface from {len(df)} scatter points "
            f"(need at least {_MIN_RBF_POINTS})"
        )

    pts = df[["log_moneyness", "T"]].to_numpy()
    vals = df["implied_volatility"].to_numpy()

    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-3)

    KK, TT = np.meshgrid(k_grid, tenor_grid, indexing="ij")
    query = np.column_stack([KK.ravel(), TT.ravel()])
    iv_flat = rbf(query)
    iv_grid = iv_flat.reshape(len(k_grid), len(tenor_grid))

    return k_grid, tenor_grid, iv_grid
