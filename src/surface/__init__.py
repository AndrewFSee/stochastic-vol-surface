"""Surface construction sub-package.

Public API
----------
VolSurface           – queryable volatility surface
build_surface_grid   – raw chain → scatter DataFrame
interpolate_to_grid  – scatter → regular (k, T) grid
build_one            – one stored chain → persisted surface
build_corpus         – sweep the whole options store
load_surface_history – stacked surface array for model training
"""

from src.surface.batch import (
    build_corpus,
    build_one,
    load_surface_history,
    load_surfaces,
    surface_path,
)
from src.surface.grid_builder import (
    build_surface_grid,
    default_k_grid,
    default_t_grid,
    interpolate_to_grid,
    load_grid_config,
)
from src.surface.surface import VolSurface

__all__ = [
    "VolSurface",
    "build_surface_grid",
    "interpolate_to_grid",
    "load_grid_config",
    "default_k_grid",
    "default_t_grid",
    "build_one",
    "build_corpus",
    "load_surface_history",
    "load_surfaces",
    "surface_path",
]
