"""
Transform a raw options chain into a standardised log-moneyness × tenor grid.

The grid uses:
- **x-axis**: log-moneyness  k = log(K / F)  where F = S * exp(r * T)
- **y-axis**: tenor in years T

Each cell contains the implied volatility σ(k, T).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.surface.implied_vol import implied_vol

logger = logging.getLogger(__name__)


def build_grid(
    chain: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    moneyness_grid: np.ndarray,
    tenor_grid: np.ndarray,
) -> pd.DataFrame:
    """Convert a raw options chain to a standardised IV surface grid.

    Parameters
    ----------
    chain:
        Options chain DataFrame with columns: ``strike``, ``days_to_expiry``,
        ``option_type``, ``mid_price``, ``bid``, ``ask``.
    spot:
        Current spot price of the underlying.
    risk_free_rate:
        Continuously-compounded risk-free rate (flat for now).
    moneyness_grid:
        1-D array of log-moneyness values (e.g. ``np.arange(-0.20, 0.21, 0.01)``).
    tenor_grid:
        1-D array of tenors in years (e.g. ``[1/52, 1/12, 3/12, 6/12, 1.0]``).

    Returns
    -------
    pd.DataFrame
        Grid DataFrame with shape ``(len(moneyness_grid), len(tenor_grid))``.
        Index = moneyness values, columns = tenor values (in years).
        Values are implied vols; ``NaN`` where no data / failed inversion.
    """
    grid = pd.DataFrame(
        index=np.round(moneyness_grid, 4),
        columns=np.round(tenor_grid, 6),
        dtype=float,
    )
    grid[:] = np.nan
    grid.index.name = "log_moneyness"
    grid.columns.name = "tenor_years"

    # Group chain by expiry tenor
    for dte, group in chain.groupby("days_to_expiry"):
        T = dte / 365.0
        # Find the closest tenor in the grid
        if len(tenor_grid) == 0:
            continue
        nearest_T = tenor_grid[np.argmin(np.abs(tenor_grid - T))]
        if abs(nearest_T - T) > 14 / 365.0:  # Skip if >2 weeks from grid point
            continue

        # Find the exact column key in the grid (handles float precision)
        col_idx = int(np.argmin(np.abs(np.array(grid.columns.astype(float)) - nearest_T)))
        col_key = grid.columns[col_idx]

        F = spot * np.exp(risk_free_rate * T)

        for _, row in group.iterrows():
            K = row["strike"]
            if K <= 0 or F <= 0:
                continue

            k = np.log(K / F)

            # Find nearest moneyness bin
            nearest_k = moneyness_grid[np.argmin(np.abs(moneyness_grid - k))]
            if abs(nearest_k - k) > 0.015:  # Skip if >1.5% from grid point
                continue

            # Prefer mid-price; fall back to last price
            price = row.get("mid_price", np.nan)
            if not np.isfinite(price) or price <= 0:
                price = row.get("last_price", np.nan)
            if not np.isfinite(price) or price <= 0:
                continue

            opt_type = str(row.get("option_type", "call"))
            iv = implied_vol(float(price), spot, K, T, risk_free_rate, opt_type)
            if iv is not None and np.isfinite(iv):
                # Find the exact row key in the grid
                row_idx = int(np.argmin(np.abs(np.array(grid.index.astype(float)) - nearest_k)))
                row_key = grid.index[row_idx]
                # Use put IV for OTM puts, call IV for OTM calls
                existing = grid.loc[row_key, col_key]
                if np.isnan(existing):
                    grid.loc[row_key, col_key] = iv
                else:
                    # Average duplicate mappings
                    grid.loc[row_key, col_key] = (existing + iv) / 2.0

    return grid


def compute_log_moneyness(
    chain: pd.DataFrame,
    spot: float,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Append a ``log_moneyness`` column to a chain DataFrame.

    Uses the forward price F = S * exp(r * T) so that k = log(K/F).
    """
    chain = chain.copy()
    T = chain["days_to_expiry"] / 365.0
    F = spot * np.exp(risk_free_rate * T)
    chain["forward_price"] = F
    chain["log_moneyness"] = np.log(chain["strike"] / F)
    return chain
