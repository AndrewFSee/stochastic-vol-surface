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
    r : float            – reference (3M) risk-free rate for the snapshot
    observed : np.ndarray or None – bool (n_k, n_t); True where the cell lies
        inside the quoted strikes of the bracketing expiries, False where it
        is extrapolated.  None for surfaces built before this was recorded.
    slices : list[dict]  – per-expiry fit diagnostics (forward, SVI params,
        quoted k-range, fit RMSE); empty for legacy surfaces.
    builder : str        – construction method/version that produced the grid
    """

    ticker: str
    as_of: date
    k_grid: np.ndarray
    t_grid: np.ndarray
    iv_grid: np.ndarray
    spot: float = 100.0
    r: float = 0.05
    observed: Optional[np.ndarray] = None
    slices: list = field(default_factory=list)
    builder: str = "unknown"
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

    def delta_strike(self, delta: float, tenor: float, n_iter: int = 8) -> float:
        """Forward log-moneyness of the option with forward delta *delta*.

        Positive *delta* is a call (e.g. ``0.25``), negative a put
        (``-0.25``).  Solves ``k = -N^{-1}(Δc)·σ√T + σ²T/2``, where ``Δc`` is
        the call-equivalent delta, iterating because σ is the smile vol at
        the very strike being solved for.
        """
        from scipy.stats import norm

        call_delta = delta if delta > 0 else 1.0 + delta
        z = norm.ppf(call_delta)
        sig = self.atm_vol(tenor)
        k = 0.0
        for _ in range(n_iter):
            k = -z * sig * np.sqrt(tenor) + 0.5 * sig ** 2 * tenor
            sig = float(self.iv(k, tenor))
        return float(k)

    def skew(self, tenor: float, delta: float = 0.25) -> float:
        """Put-minus-call implied vol at ±*delta* (default 25-delta).

        This is the negative of the conventional risk reversal.
        """
        k_put = self.delta_strike(-delta, tenor)
        k_call = self.delta_strike(delta, tenor)
        return float(self.iv(k_put, tenor)) - float(self.iv(k_call, tenor))

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
                row = {"log_moneyness": k, "tenor": T, "implied_vol": self.iv_grid[ki, ti]}
                if self.observed is not None:
                    row["observed"] = bool(self.observed[ki, ti])
                rows.append(row)
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
        rate_fn: Optional[Callable[[float], float]] = None,
        apply_calendar_filter: bool = True,
        k_grid: np.ndarray | None = None,
        t_grid: np.ndarray | None = None,
    ) -> VolSurface:
        """Build a VolSurface from a raw options chain DataFrame.

        Per-expiry construction (see :mod:`src.surface.slices`):

        1. two-sided quotes → forward from put-call parity per expiry
        2. OTM Black-76 implied vols from the mids
        3. weighted, constrained SVI fit per expiry
        4. total variance linear in T between bracketing expiries
        5. ``apply_calendar_filter_grid`` – enforce total-variance monotonicity

        Parameters
        ----------
        chain : pd.DataFrame
            Raw options chain (needs strike, T, option_type, bid/ask or mid,
            or a quoted IV column for price-less historical sessions).
        ticker, as_of, spot : identification & market parameters.
        r : float
            Reference rate stored on the surface; also the discount rate for
            every expiry unless *rate_fn* is given.
        rate_fn : callable ``T -> r``, optional
            Term structure of rates, used per expiry for discounting.
        apply_calendar_filter : bool
            Enforce non-decreasing total variance across tenors on the grid.
        k_grid, t_grid : optional grid override (defaults from config).
        """
        from src.surface.filters import apply_calendar_filter_grid
        from src.surface.grid import default_k_grid, default_t_grid
        from src.surface.slices import BUILDER_VERSION, fit_slices, slices_to_grid

        k_nodes = default_k_grid() if k_grid is None else np.asarray(k_grid, float)
        t_nodes = default_t_grid() if t_grid is None else np.asarray(t_grid, float)
        fits = fit_slices(chain, spot, rate_fn if rate_fn is not None else r)
        if len(fits) < 2:
            raise ValueError(f"only {len(fits)} expiries could be fitted (need 2)")
        iv_grid, observed = slices_to_grid(fits, k_nodes, t_nodes)
        if apply_calendar_filter:
            iv_grid = apply_calendar_filter_grid(k_nodes, t_nodes, iv_grid)
        return cls(
            ticker=ticker, as_of=as_of, k_grid=k_nodes, t_grid=t_nodes,
            iv_grid=iv_grid, spot=spot, r=r, observed=observed,
            slices=[f.to_dict() for f in fits], builder=BUILDER_VERSION,
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
            "builder": self.builder,
            "slices": self.slices,
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

        observed = None
        if "observed" in df.columns:
            observed = (
                df.pivot_table(index="log_moneyness", columns="tenor",
                               values="observed", aggfunc="max")
                .reindex(index=k_grid, columns=t_grid)
                .fillna(False).to_numpy(dtype=bool)
            )

        return cls(
            ticker=meta["ticker"],
            as_of=date.fromisoformat(meta["as_of"]),
            k_grid=k_grid,
            t_grid=t_grid,
            iv_grid=iv_grid,
            spot=meta["spot"],
            r=meta.get("r", 0.05),
            observed=observed,
            slices=meta.get("slices", []),
            # Surfaces written before the builder was recorded came from the
            # pooled-bin pipeline.
            builder=meta.get("builder", "legacy-svi"),
        )
