"""VolSurface: a queryable volatility surface backed by RegularGridInterpolator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class VolSurface:
    """Arbitrage-free implied volatility surface.

    Attributes
    ----------
    ticker : str
    as_of : date
    k_grid : np.ndarray  – log-moneyness knots, shape (n_k,)
    t_grid : np.ndarray  – tenor knots in years, shape (n_t,)
    iv_grid : np.ndarray – implied vol values, shape (n_k, n_t)
    """

    ticker: str
    as_of: date
    k_grid: np.ndarray
    t_grid: np.ndarray
    iv_grid: np.ndarray
    spot: float = 100.0
    _interpolator: Optional[RegularGridInterpolator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._interpolator = RegularGridInterpolator(
            (self.k_grid, self.t_grid),
            self.iv_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate
        )

    def iv(self, log_moneyness: float | np.ndarray, tenor: float | np.ndarray) -> np.ndarray:
        """Query implied volatility for scalar or array (log_moneyness, tenor) pairs."""
        pts = np.column_stack(
            [np.atleast_1d(log_moneyness), np.atleast_1d(tenor)]
        )
        return self._interpolator(pts).squeeze()

    def atm_vol(self, tenor: float) -> float:
        """ATM (log-moneyness=0) implied vol for a given tenor."""
        return float(self.iv(0.0, tenor))

    def skew(self, tenor: float, delta: float = 0.25) -> float:
        """Approximate 25-delta put-call skew for a given tenor.

        Uses log-moneyness ≈ ±0.25*sigma*sqrt(T) as the 25-delta strikes.
        """
        atm = self.atm_vol(tenor)
        dk = delta * atm * np.sqrt(tenor)
        call_iv = float(self.iv(dk, tenor))
        put_iv = float(self.iv(-dk, tenor))
        return put_iv - call_iv

    def term_structure(self) -> np.ndarray:
        """ATM vol term structure over the defined tenor grid."""
        return np.array([self.atm_vol(T) for T in self.t_grid])

    def to_dataframe(self) -> "import pandas; pandas.DataFrame":
        """Export the full grid as a long-format DataFrame."""
        import pandas as pd

        rows = []
        for ki, k in enumerate(self.k_grid):
            for ti, T in enumerate(self.t_grid):
                rows.append({"log_moneyness": k, "tenor": T, "implied_vol": self.iv_grid[ki, ti]})
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: "import pandas; pandas.DataFrame",
        ticker: str,
        as_of: date,
        k_grid: np.ndarray,
        t_grid: np.ndarray,
        spot: float = 100.0,
    ) -> "VolSurface":
        """Construct a VolSurface by pivoting a long-format DataFrame onto the grid."""
        import pandas as pd

        pivot = df.pivot_table(
            index="log_moneyness", columns="tenor", values="implied_vol", aggfunc="mean"
        )
        iv_grid = pivot.reindex(index=k_grid, columns=t_grid).to_numpy()
        return cls(
            ticker=ticker,
            as_of=as_of,
            k_grid=k_grid,
            t_grid=t_grid,
            iv_grid=iv_grid,
            spot=spot,
        )
