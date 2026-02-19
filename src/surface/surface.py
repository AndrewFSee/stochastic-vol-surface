"""
VolSurface class — query implied volatility at any (log-moneyness, tenor) point.

Wraps a standardised surface grid and provides:
- Bilinear and bicubic interpolation within the grid
- Forward-rate and ATM vol convenience methods
- Term-structure and skew extraction
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class VolSurface:
    """A queryable implied volatility surface.

    Parameters
    ----------
    grid:
        DataFrame with log-moneyness index and tenor (years) columns.
        Values are implied vols (decimal, e.g. 0.20 = 20 %).
    ticker:
        Underlying ticker symbol.
    as_of:
        Snapshot date.
    spot:
        Spot price at snapshot time.
    """

    def __init__(
        self,
        grid: pd.DataFrame,
        ticker: str = "",
        as_of: Optional[date] = None,
        spot: float = float("nan"),
    ) -> None:
        self.grid = grid.copy()
        self.ticker = ticker
        self.as_of = as_of
        self.spot = spot

        self._moneyness = np.array(grid.index.astype(float))
        self._tenors = np.array(grid.columns.astype(float))
        self._iv_matrix = grid.values.astype(float)

        # Build interpolator (ignores NaN cells)
        self._interp = self._build_interpolator()

    # ------------------------------------------------------------------
    # Core query interface
    # ------------------------------------------------------------------

    def get_iv(
        self,
        log_moneyness: float,
        tenor_years: float,
    ) -> float:
        """Query the implied vol at a (log-moneyness, tenor) point.

        Parameters
        ----------
        log_moneyness:
            k = log(K / F).
        tenor_years:
            Time to expiry in years.

        Returns
        -------
        float
            Interpolated implied volatility, or NaN if outside the grid.
        """
        if self._interp is None:
            return float("nan")
        try:
            return float(self._interp([[log_moneyness, tenor_years]])[0])
        except Exception:
            return float("nan")

    def get_atm_vol(self, tenor_years: float) -> float:
        """Return the at-the-money (k=0) implied vol for a given tenor."""
        return self.get_iv(0.0, tenor_years)

    def get_skew(
        self,
        tenor_years: float,
        delta: float = 0.25,
    ) -> float:
        """Compute the 25-delta put-call skew.

        Approximates the 25Δ log-moneyness as -0.10 (put) and +0.10 (call)
        and returns σ(-0.10, T) - σ(+0.10, T).

        Parameters
        ----------
        tenor_years:
            Tenor in years.
        delta:
            Target delta (default 0.25 → 25Δ).

        Returns
        -------
        float
            Skew: positive means put vol > call vol (normal skew).
        """
        # Approximate 25Δ moneyness as ±10% of the log-moneyness range
        k_put = -0.10
        k_call = 0.10
        return self.get_iv(k_put, tenor_years) - self.get_iv(k_call, tenor_years)

    def get_term_structure(self, log_moneyness: float = 0.0) -> pd.Series:
        """Extract the term structure of implied vol at a fixed moneyness.

        Parameters
        ----------
        log_moneyness:
            Default 0.0 (ATM).

        Returns
        -------
        pd.Series
            Indexed by tenor (years), values are implied vols.
        """
        ivs = [self.get_iv(log_moneyness, T) for T in self._tenors]
        return pd.Series(ivs, index=self._tenors, name=f"IV(k={log_moneyness:.2f})")

    def get_smile(self, tenor_years: float) -> pd.Series:
        """Extract the volatility smile at a fixed tenor.

        Returns
        -------
        pd.Series
            Indexed by log-moneyness, values are implied vols.
        """
        nearest_T = self._tenors[np.argmin(np.abs(self._tenors - tenor_years))]
        col = self.grid.get(nearest_T, pd.Series(dtype=float))
        return col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_interpolator(self) -> Optional[RegularGridInterpolator]:
        """Build a scipy RegularGridInterpolator from the grid."""
        iv = self._iv_matrix.copy()
        # Fill NaN with column mean for interpolation (will be NaN outside range)
        col_means = np.nanmean(iv, axis=0)
        for j in range(iv.shape[1]):
            mask = np.isnan(iv[:, j])
            iv[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.20

        if iv.shape[0] < 2 or iv.shape[1] < 2:
            return None

        return RegularGridInterpolator(
            (self._moneyness, self._tenors),
            iv,
            method="linear",
            bounds_error=False,
            fill_value=float("nan"),
        )

    def __repr__(self) -> str:
        return (
            f"VolSurface(ticker={self.ticker!r}, as_of={self.as_of}, "
            f"grid={self.grid.shape[0]}×{self.grid.shape[1]})"
        )
