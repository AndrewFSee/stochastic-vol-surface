"""Convert a raw options chain DataFrame into a standardised log-moneyness × tenor grid."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from src.surface.implied_vol import implied_vol_vectorized

logger = logging.getLogger(__name__)


def build_surface_grid(
    chain: pd.DataFrame,
    spot: float,
    r: float = 0.05,
    moneyness_range: tuple[float, float] = (-0.40, 0.20),
    n_moneyness: int = 25,
    min_T: float = 1 / 365,
    min_open_interest: int = 0,
) -> pd.DataFrame:
    """Convert a raw options chain to a (log_moneyness, tenor, implied_vol) DataFrame.

    Parameters
    ----------
    chain : pd.DataFrame
        Must have columns: strike, T, option_type, bid, ask (or mid or last_price),
        open_interest (optional).
    spot : float
        Current underlying spot price.
    r : float
        Risk-free rate (annualised).
    moneyness_range : tuple
        (min, max) log-moneyness filter.
    n_moneyness : int
        Number of grid points in log-moneyness dimension.
    min_T : float
        Minimum time-to-expiry (years) to include.
    min_open_interest : int
        Minimum open interest filter.
    """
    df = chain.copy()

    # Filter expiry
    if "T" not in df.columns:
        raise ValueError("chain must have column 'T' (time-to-expiry in years)")
    df = df[df["T"] > min_T].copy()

    # Filter open interest
    if "open_interest" in df.columns and min_open_interest > 0:
        df = df[df["open_interest"] >= min_open_interest]

    # Compute mid price
    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        elif "last_price" in df.columns:
            df["mid"] = df["last_price"]
        else:
            raise ValueError("chain must have 'mid', 'bid'/'ask', or 'last_price'")

    df = df.dropna(subset=["mid", "strike", "T", "option_type"])
    df = df[df["mid"] > 0]

    # Compute IV
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

    # Compute log-moneyness
    df["log_moneyness"] = np.log(df["strike"] / spot)

    # Filter moneyness range
    df = df[
        (df["log_moneyness"] >= moneyness_range[0])
        & (df["log_moneyness"] <= moneyness_range[1])
    ]

    return df[["log_moneyness", "T", "implied_volatility", "option_type", "strike"]].reset_index(
        drop=True
    )


def interpolate_to_grid(
    surface_df: pd.DataFrame,
    k_grid: Optional[np.ndarray] = None,
    tenor_grid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin-plate spline interpolation onto a regular (k, T) grid.

    Returns
    -------
    k_nodes : np.ndarray  shape (n_k,)
    t_nodes : np.ndarray  shape (n_t,)
    iv_grid : np.ndarray  shape (n_k, n_t) – NaN where extrapolated
    """
    from scipy.interpolate import RBFInterpolator

    if k_grid is None:
        k_grid = np.linspace(-0.40, 0.20, 25)
    if tenor_grid is None:
        tenor_grid = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])

    pts = surface_df[["log_moneyness", "T"]].to_numpy()
    vals = surface_df["implied_volatility"].to_numpy()

    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-3)

    KK, TT = np.meshgrid(k_grid, tenor_grid, indexing="ij")
    query = np.column_stack([KK.ravel(), TT.ravel()])
    iv_flat = rbf(query)
    iv_grid = iv_flat.reshape(len(k_grid), len(tenor_grid))

    return k_grid, tenor_grid, iv_grid
