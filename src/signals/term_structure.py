"""
Term structure of implied volatility: slope, curvature, and inversion detection.

The term structure encodes the market's expectation of future realised vol
across different horizons.  Key signals:
- **Normal**: Short-dated IV < Long-dated IV (upward sloping)
- **Inverted**: Short-dated IV > Long-dated IV (market stress)
- **Hump**: Mid-dated IV peaks above both ends (event risk premia)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_term_structure_slope(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    short_tenor: float = 1 / 12,
    long_tenor: float = 6 / 12,
    log_moneyness: float = 0.0,
) -> float:
    """Compute the ATM term structure slope between two tenors.

    slope = σ(k=0, T_long) - σ(k=0, T_short)

    Positive = upward sloping (normal).
    Negative = inverted (stress signal).

    Parameters
    ----------
    surface:
        VolSurface object.
    short_tenor, long_tenor:
        Tenor endpoints in years.
    log_moneyness:
        Moneyness to compute at (default 0 = ATM).

    Returns
    -------
    float
        Term structure slope (annualised vol points).
    """
    iv_short = surface.get_iv(log_moneyness, short_tenor)
    iv_long = surface.get_iv(log_moneyness, long_tenor)
    return float(iv_long - iv_short)


def compute_term_structure_curvature(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    short_tenor: float = 1 / 12,
    mid_tenor: float = 3 / 12,
    long_tenor: float = 12 / 12,
    log_moneyness: float = 0.0,
) -> float:
    """Compute the ATM term structure curvature (butterfly across tenors).

    curvature = σ_short + σ_long - 2 * σ_mid

    Positive = humped term structure (event risk premia in the middle).
    Negative = U-shaped (unusual).

    Returns
    -------
    float
        Term structure curvature.
    """
    iv_s = surface.get_iv(log_moneyness, short_tenor)
    iv_m = surface.get_iv(log_moneyness, mid_tenor)
    iv_l = surface.get_iv(log_moneyness, long_tenor)
    return float(iv_s + iv_l - 2 * iv_m)


def detect_inversion(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    tenors_years: list[float] | None = None,
    threshold: float = -0.02,
    log_moneyness: float = 0.0,
) -> bool:
    """Detect term structure inversion.

    Returns True if the ATM term structure slope between the shortest and
    longest tenors is below ``threshold``.

    Parameters
    ----------
    threshold:
        Inversion threshold in vol units (default -2 vol points).
    """
    if tenors_years is None:
        tenors_years = [1 / 52, 1 / 12, 3 / 12, 6 / 12, 1.0]

    iv_short = surface.get_iv(log_moneyness, tenors_years[0])
    iv_long = surface.get_iv(log_moneyness, tenors_years[-1])
    slope = iv_long - iv_short
    return float(slope) < threshold


def compute_term_structure_timeseries(
    surfaces: list,
    dates: list,
    short_tenor: float = 1 / 12,
    long_tenor: float = 6 / 12,
) -> pd.DataFrame:
    """Compute term structure slope / curvature time series.

    Returns
    -------
    pd.DataFrame
        Columns: ``slope``, ``inverted``.
    """
    slopes, inverted_flags = [], []
    for surf in surfaces:
        slope = compute_term_structure_slope(surf, short_tenor, long_tenor)
        slopes.append(slope)
        inverted_flags.append(detect_inversion(surf))

    return pd.DataFrame({
        "slope": slopes,
        "inverted": inverted_flags,
    }, index=dates)


def term_structure_signal(
    slope_series: pd.Series,
    inversion_threshold: float = -0.02,
    lookback: int = 63,
) -> pd.DataFrame:
    """Generate term structure trading signals.

    - Signal +1 (sell front/buy back) when slope is historically flat/inverted.
    - Signal -1 (buy front/sell back) when slope is historically steep.
    - Signal 0 (neutral) otherwise.

    Returns
    -------
    pd.DataFrame
        Columns: ``slope``, ``zscore``, ``signal``.
    """
    roll_mean = slope_series.rolling(lookback, min_periods=20).mean()
    roll_std = slope_series.rolling(lookback, min_periods=20).std()
    zscore = (slope_series - roll_mean) / roll_std.replace(0, float("nan"))

    signal = np.where(slope_series < inversion_threshold, 1,
                      np.where(zscore > 1.5, -1, 0))
    return pd.DataFrame({
        "slope": slope_series,
        "zscore": zscore,
        "signal": pd.Series(signal, index=slope_series.index),
    })
