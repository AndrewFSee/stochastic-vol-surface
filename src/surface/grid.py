"""The standard (log-moneyness × tenor) grid stored with every surface.

Defined in ``config/surface_grid.yaml``.  The grid is a fixed-shape summary
for storage and quick plotting; features are read from the per-expiry fits
instead (see :mod:`src.surface.slices`), so they do not depend on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "surface_grid.yaml"


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
