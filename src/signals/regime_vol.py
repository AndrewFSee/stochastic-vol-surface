"""Vol regime classifier: Low / Normal / High / Crisis."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class VolRegime(str, Enum):
    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"
    CRISIS = "Crisis"


_REGIME_THRESHOLDS = {
    "low_vix": 15.0,
    "high_vix": 25.0,
    "crisis_vix": 35.0,
}


def classify_regime(
    vix: float,
    low_thresh: float = 15.0,
    high_thresh: float = 25.0,
    crisis_thresh: float = 35.0,
) -> VolRegime:
    """Classify vol regime based on VIX level."""
    if vix >= crisis_thresh:
        return VolRegime.CRISIS
    elif vix >= high_thresh:
        return VolRegime.HIGH
    elif vix <= low_thresh:
        return VolRegime.LOW
    return VolRegime.NORMAL


def classify_regime_hmm(
    vix_series: np.ndarray,
    n_states: int = 4,
) -> np.ndarray:
    """Classify vol regimes using a hidden Markov model (requires hmmlearn).

    Falls back to threshold-based classification if hmmlearn is not available.

    Returns
    -------
    labels : np.ndarray of VolRegime values, same length as vix_series
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        log_vix = np.log(vix_series).reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        model.fit(log_vix)
        states = model.predict(log_vix)
        # Map states to regimes by mean VIX level
        means = [np.exp(model.means_[s, 0]) for s in range(n_states)]
        sorted_states = np.argsort(means)
        regime_map = {
            int(sorted_states[0]): VolRegime.LOW,
            int(sorted_states[1]): VolRegime.NORMAL,
            int(sorted_states[2]): VolRegime.HIGH,
            int(sorted_states[3]): VolRegime.CRISIS,
        }
        return np.array([regime_map[int(s)] for s in states])
    except ImportError:
        return np.array([classify_regime(v) for v in vix_series])


def regime_time_series(vix_series: pd.Series) -> pd.Series:
    """Apply threshold-based regime classification to a VIX time series."""
    return vix_series.map(classify_regime)
