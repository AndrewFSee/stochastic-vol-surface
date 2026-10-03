"""25-delta put-call skew steepening/flattening detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class SkewSignal(str, Enum):
    STEEPENING = "STEEPENING"   # put skew increasing (bearish)
    FLATTENING = "FLATTENING"   # put skew decreasing (bullish)
    NEUTRAL = "NEUTRAL"


@dataclass
class SkewSnapshot:
    tenor: float          # years
    put_25d_iv: float
    atm_iv: float
    call_25d_iv: float

    @property
    def put_call_skew(self) -> float:
        """25-delta put-call skew: put_iv - call_iv."""
        return self.put_25d_iv - self.call_25d_iv

    @property
    def risk_reversal(self) -> float:
        """25-delta risk reversal: call_iv - put_iv."""
        return self.call_25d_iv - self.put_25d_iv

    @property
    def butterfly(self) -> float:
        """25-delta butterfly: (put_iv + call_iv)/2 - atm_iv."""
        return (self.put_25d_iv + self.call_25d_iv) / 2 - self.atm_iv


def compute_skew_signal(
    current_skew: float,
    historical_skews: np.ndarray,
    z_threshold: float = 1.5,
) -> SkewSignal:
    """Return skew signal based on z-score of current skew vs history.

    Positive z-score (skew above average) = steepening.
    Negative z-score = flattening.
    """
    if len(historical_skews) < 2:
        return SkewSignal.NEUTRAL

    mu = float(np.mean(historical_skews))
    sigma = float(np.std(historical_skews, ddof=1))
    if sigma < 1e-10:
        return SkewSignal.NEUTRAL

    z = (current_skew - mu) / sigma
    if z > z_threshold:
        return SkewSignal.STEEPENING
    elif z < -z_threshold:
        return SkewSignal.FLATTENING
    return SkewSignal.NEUTRAL


def skew_time_series(
    surface_history: list[dict],
    tenor: float = 0.25,
) -> pd.DataFrame:
    """Extract skew time series from a list of surface dicts.

    Each dict must have keys: date, put_25d_iv, atm_iv, call_25d_iv (for the given tenor).
    """
    rows = []
    for snap in surface_history:
        rows.append({
            "date": snap.get("date"),
            "put_call_skew": snap.get("put_25d_iv", np.nan) - snap.get("call_25d_iv", np.nan),
            "risk_reversal": snap.get("call_25d_iv", np.nan) - snap.get("put_25d_iv", np.nan),
            "butterfly": (snap.get("put_25d_iv", np.nan) + snap.get("call_25d_iv", np.nan)) / 2
                         - snap.get("atm_iv", np.nan),
        })
    return pd.DataFrame(rows).set_index("date")
