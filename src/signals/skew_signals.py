"""
25-delta put-call skew steepening / flattening detection signals.

The skew signal measures the difference between OTM put and OTM call
implied vols as a proxy for demand imbalance and tail-risk sentiment.

A steep negative skew (put IV >> call IV) signals elevated crash risk;
a flat or positive skew (unusual) may signal complacency or forced call buying.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate 25-delta log-moneyness (depends on vol and tenor but this is typical)
_K_PUT_25D = -0.10   # ~25Δ OTM put log-moneyness
_K_CALL_25D = 0.10   # ~25Δ OTM call log-moneyness


def compute_skew(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    tenor_years: float,
) -> float:
    """Compute 25-delta put-call skew for a given tenor.

    Parameters
    ----------
    surface:
        VolSurface object.
    tenor_years:
        Target tenor in years.

    Returns
    -------
    float
        Skew = σ(k=-0.10, T) - σ(k=+0.10, T).
        Positive → put IV > call IV (normal downside skew).
    """
    iv_put = surface.get_iv(_K_PUT_25D, tenor_years)
    iv_call = surface.get_iv(_K_CALL_25D, tenor_years)
    return float(iv_put - iv_call)


def compute_skew_timeseries(
    surfaces: list,
    dates: list,
    tenor_years: float = 1 / 12,
) -> pd.Series:
    """Compute skew time series from a list of VolSurface snapshots.

    Parameters
    ----------
    surfaces:
        List of VolSurface objects ordered by date.
    dates:
        Corresponding dates.
    tenor_years:
        Tenor to compute skew at.

    Returns
    -------
    pd.Series
        Skew time series indexed by date.
    """
    skews = [compute_skew(s, tenor_years) for s in surfaces]
    return pd.Series(skews, index=dates, name=f"skew_25d_{tenor_years:.2f}y")


def skew_zscore(
    skew_series: pd.Series,
    lookback: int = 63,
) -> pd.Series:
    """Compute rolling Z-score of the skew series.

    Parameters
    ----------
    skew_series:
        Time series of 25-delta skew values.
    lookback:
        Rolling window in trading days (default 63 ≈ 3 months).

    Returns
    -------
    pd.Series
        Rolling Z-score.
    """
    roll_mean = skew_series.rolling(lookback, min_periods=20).mean()
    roll_std = skew_series.rolling(lookback, min_periods=20).std()
    return (skew_series - roll_mean) / roll_std.replace(0, float("nan"))


def detect_skew_signals(
    skew_series: pd.Series,
    zscore_threshold: float = 1.5,
    lookback: int = 63,
) -> pd.DataFrame:
    """Detect skew steepening / flattening signals.

    Signals are generated when the Z-score of the skew crosses the threshold:
    - **Steep** (Z > +threshold): Skew is historically wide → sell skew.
    - **Flat** (Z < -threshold): Skew is historically narrow → buy skew.

    Parameters
    ----------
    skew_series:
        Time series of 25-delta skew.
    zscore_threshold:
        Signal threshold (default 1.5σ).
    lookback:
        Z-score rolling window.

    Returns
    -------
    pd.DataFrame
        Columns: ``skew``, ``zscore``, ``signal``
        where signal ∈ {-1 (flat/buy), 0 (neutral), +1 (steep/sell)}.
    """
    z = skew_zscore(skew_series, lookback)
    signal = np.where(z > zscore_threshold, 1, np.where(z < -zscore_threshold, -1, 0))
    return pd.DataFrame({
        "skew": skew_series,
        "zscore": z,
        "signal": pd.Series(signal, index=skew_series.index),
    })
