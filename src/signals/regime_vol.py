"""
Volatility regime classifier.

Classifies the current market environment into:
- **Low** (VIX < 15)
- **Normal** (15 ≤ VIX < 25)
- **High** (25 ≤ VIX < 40)
- **Crisis** (VIX ≥ 40)

Also considers the vol surface *shape* (skew steepness, term structure slope)
to refine the regime classification.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.vix_family import get_vix_zscore, classify_regime

logger = logging.getLogger(__name__)

REGIME_LABELS = ["Low", "Normal", "High", "Crisis"]


def classify_vol_regime(
    vix_level: float,
    vix_zscore: Optional[float] = None,
    skew_zscore: Optional[float] = None,
    term_slope: Optional[float] = None,
    vix_low: float = 15.0,
    vix_high: float = 25.0,
    vix_crisis: float = 40.0,
) -> str:
    """Classify the current vol regime.

    Uses the VIX level as the primary signal with optional adjustments from
    the vol surface shape.

    Parameters
    ----------
    vix_level:
        Current VIX closing level.
    vix_zscore:
        Rolling Z-score of VIX (optional).
    skew_zscore:
        Rolling Z-score of the 25Δ skew (optional).
    term_slope:
        ATM term structure slope (optional).
    vix_low, vix_high, vix_crisis:
        VIX thresholds for regime boundaries.

    Returns
    -------
    str
        Regime label: ``"Low"``, ``"Normal"``, ``"High"``, or ``"Crisis"``.
    """
    regime = classify_regime(vix_level, vix_low, vix_high, vix_crisis)

    # Upgrade regime based on surface signals
    if regime == "Normal" and vix_zscore is not None and vix_zscore > 2.0:
        regime = "High"
    if regime == "High" and term_slope is not None and term_slope < -0.05:
        regime = "Crisis"  # Deep inversion + high VIX = crisis

    return regime


def build_regime_series(
    vix_df: pd.DataFrame,
    surfaces: Optional[list] = None,
    surface_dates: Optional[list] = None,
    lookback: int = 252,
) -> pd.DataFrame:
    """Build a daily vol-regime classification time series.

    Parameters
    ----------
    vix_df:
        DataFrame with a ``VIX`` column indexed by date.
    surfaces:
        Optional list of VolSurface objects (for surface-shape adjustments).
    surface_dates:
        Dates corresponding to ``surfaces``.
    lookback:
        Rolling window for VIX Z-score.

    Returns
    -------
    pd.DataFrame
        Columns: ``vix``, ``vix_zscore``, ``regime``.
    """
    vix_series = vix_df["VIX"].dropna()
    vix_z = get_vix_zscore(vix_series, lookback)

    regimes = []
    for dt, vix_val in vix_series.items():
        z = vix_z.get(dt, float("nan"))
        regime = classify_vol_regime(
            vix_level=float(vix_val),
            vix_zscore=float(z) if np.isfinite(z) else None,
        )
        regimes.append(regime)

    result = pd.DataFrame({
        "vix": vix_series,
        "vix_zscore": vix_z,
        "regime": pd.Series(regimes, index=vix_series.index),
    })
    return result


def regime_adjusted_position_size(
    base_size: float,
    regime: str,
    regime_scalars: Optional[dict[str, float]] = None,
) -> float:
    """Scale position size based on vol regime.

    In high-vol regimes, reduce position sizes to manage tail risk.

    Parameters
    ----------
    base_size:
        Nominal position size.
    regime:
        Current regime label.
    regime_scalars:
        Override dict mapping regime → size scalar.

    Returns
    -------
    float
        Scaled position size.
    """
    defaults: dict[str, float] = {
        "Low": 1.0,
        "Normal": 0.75,
        "High": 0.50,
        "Crisis": 0.25,
    }
    scalars = regime_scalars or defaults
    return base_size * scalars.get(regime, 0.5)
