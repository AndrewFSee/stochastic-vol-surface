"""VolSurface: a queryable volatility surface backed by RegularGridInterpolator.

Constructors
------------
- ``VolSurface(...)``          – from pre-computed grids
- ``VolSurface.from_chain()``  – full pipeline: raw chain → IV surface
- ``VolSurface.from_dataframe()`` – from a long-format DataFrame
- ``VolSurface.load()``        – from a saved Parquet file

Persistence
-----------
- ``surface.save(path)``  – write to Parquet (long-format + metadata)
- ``VolSurface.load(path)`` – read back
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


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
    spot : float
    r : float            – risk-free rate used to build the surface
    """

    ticker: str
    as_of: date
    k_grid: np.ndarray
    t_grid: np.ndarray
    iv_grid: np.ndarray
    spot: float = 100.0
    r: float = 0.05
    _interpolator: Optional[RegularGridInterpolator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._interpolator = RegularGridInterpolator(
            (self.k_grid, self.t_grid),
            self.iv_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,  # nearest-neighbour extrapolation
        )

    # ── Queries ───────────────────────────────────────────────────────────

    def iv(
        self,
        log_moneyness: float | np.ndarray = 0.0,
        tenor: float | np.ndarray = 0.25,
    ) -> np.ndarray:
        """Query implied volatility at (log_moneyness, tenor) point(s)."""
        pts = np.column_stack(
            [np.atleast_1d(log_moneyness), np.atleast_1d(tenor)]
        )
        return self._interpolator(pts).squeeze()

    def atm_vol(self, tenor: float) -> float:
        """ATM (log-moneyness=0) implied vol for a given tenor."""
        return float(self.iv(0.0, tenor))

    def skew(self, tenor: float, delta: float = 0.25) -> float:
        """Approximate 25-delta put-call skew for a given tenor.

        Uses log-moneyness ≈ ±delta·σ·√T as the 25-delta strikes.
        """
        atm = self.atm_vol(tenor)
        dk = delta * atm * np.sqrt(tenor)
        call_iv = float(self.iv(dk, tenor))
        put_iv = float(self.iv(-dk, tenor))
        return put_iv - call_iv

    def term_structure(self) -> np.ndarray:
        """ATM vol term structure over the defined tenor grid."""
        return np.array([self.atm_vol(T) for T in self.t_grid])

    # ── DataFrame export / import ─────────────────────────────────────────

    def to_dataframe(self):
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
        df,
        ticker: str,
        as_of: date,
        k_grid: np.ndarray,
        t_grid: np.ndarray,
        spot: float = 100.0,
    ) -> VolSurface:
        """Construct a VolSurface by pivoting a long-format DataFrame."""
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

    # ── Full pipeline constructor ─────────────────────────────────────────

    @classmethod
    def from_chain(
        cls,
        chain,
        ticker: str,
        as_of: date,
        spot: float,
        r: float = 0.05,
        *,
        apply_butterfly_filter: bool = True,
        apply_calendar_filter: bool = True,
        method: str = "svi",
        k_grid: np.ndarray | None = None,
        t_grid: np.ndarray | None = None,
    ) -> VolSurface:
        """Build a VolSurface from a raw options chain DataFrame.

        Pipeline stages
        ---------------
        1. ``build_surface_grid``  – IV inversion, forward moneyness, filtering
        2. ``apply_arbitrage_filters`` – remove butterfly-violating strikes
        3. ``interpolate_to_grid``  – SVI-per-slice + cubic spline (or RBF)
        4. ``apply_calendar_filter_grid`` – enforce total-variance monotonicity

        Parameters
        ----------
        chain : pd.DataFrame
            Raw options chain (needs strike, T, option_type, mid/bid+ask).
        ticker, as_of, spot, r : identification & market parameters.
        apply_butterfly_filter, apply_calendar_filter : bool
            Toggle arbitrage filters.
        method : ``"svi"`` | ``"rbf"``
        k_grid, t_grid : optional grid override (defaults from config).

        Returns
        -------
        VolSurface
        """
        from src.surface.filters import apply_arbitrage_filters, apply_calendar_filter_grid
        from src.surface.grid_builder import build_surface_grid, interpolate_to_grid

        # 1. Raw chain → scatter
        scatter = build_surface_grid(chain, spot=spot, r=r)
        logger.info("Scatter: %d points after IV inversion + moneyness filter", len(scatter))

        # 2. Butterfly filter
        if apply_butterfly_filter:
            scatter = apply_arbitrage_filters(scatter, remove_butterfly=True)
            logger.info("After butterfly filter: %d points", len(scatter))

        # 3. Interpolate to regular grid
        k_nodes, t_nodes, iv_grid = interpolate_to_grid(
            scatter, k_grid=k_grid, tenor_grid=t_grid, method=method,
        )

        # 4. Calendar arbitrage clean-up
        if apply_calendar_filter:
            iv_grid = apply_calendar_filter_grid(k_nodes, t_nodes, iv_grid)

        return cls(
            ticker=ticker,
            as_of=as_of,
            k_grid=k_nodes,
            t_grid=t_nodes,
            iv_grid=iv_grid,
            spot=spot,
            r=r,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Persist the surface as a Parquet file with JSON metadata.

        The file stores the long-format grid plus metadata (ticker, as_of,
        spot, r, k_grid, t_grid dimensions) in the Parquet schema metadata.
        """
        import pandas as pd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        meta = {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "spot": float(self.spot),
            "r": float(self.r),
            "n_k": len(self.k_grid),
            "n_t": len(self.t_grid),
            "k_min": float(self.k_grid.min()),
            "k_max": float(self.k_grid.max()),
            "t_min": float(self.t_grid.min()),
            "t_max": float(self.t_grid.max()),
        }
        # Store metadata as Parquet file-level metadata
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df)
        existing_meta = table.schema.metadata or {}
        existing_meta[b"vol_surface"] = json.dumps(meta).encode()
        table = table.replace_schema_metadata(existing_meta)
        pq.write_table(table, path)
        logger.info("Saved VolSurface to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> VolSurface:
        """Load a VolSurface from a Parquet file written by :meth:`save`."""
        import pyarrow.parquet as pq

        path = Path(path)
        table = pq.read_table(path)

        raw_meta = table.schema.metadata.get(b"vol_surface")
        if raw_meta is None:
            raise ValueError(f"No vol_surface metadata in {path}")
        meta = json.loads(raw_meta)

        df = table.to_pandas()
        k_grid = np.sort(df["log_moneyness"].unique())
        t_grid = np.sort(df["tenor"].unique())

        pivot = df.pivot_table(
            index="log_moneyness", columns="tenor", values="implied_vol", aggfunc="mean"
        )
        iv_grid = pivot.reindex(index=k_grid, columns=t_grid).to_numpy()

        return cls(
            ticker=meta["ticker"],
            as_of=date.fromisoformat(meta["as_of"]),
            k_grid=k_grid,
            t_grid=t_grid,
            iv_grid=iv_grid,
            spot=meta["spot"],
            r=meta.get("r", 0.05),
        )
