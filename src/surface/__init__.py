"""Surface construction sub-package.

Public API
----------
VolSurface           – queryable volatility surface (``VolSurface.from_chain``)
fit_slices           – raw chain → per-expiry SVI fits
ExpirySurface        – continuous (k, T) surface over those fits
build_one            – one stored chain → persisted surface
build_corpus         – sweep the whole options store
load_surfaces        – stored surfaces as ``{date: VolSurface}``
load_surface_history – stacked surface grids
"""

from src.surface.batch import (
    build_corpus,
    build_one,
    load_surface_history,
    load_surfaces,
    surface_path,
)
from src.surface.grid import default_k_grid, default_t_grid, load_grid_config
from src.surface.slices import ExpirySurface, fit_slices
from src.surface.surface import VolSurface

__all__ = [
    "ExpirySurface",
    "VolSurface",
    "build_corpus",
    "build_one",
    "default_k_grid",
    "default_t_grid",
    "fit_slices",
    "load_grid_config",
    "load_surface_history",
    "load_surfaces",
    "surface_path",
]
