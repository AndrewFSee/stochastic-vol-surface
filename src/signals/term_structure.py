"""Term structure slope, curvature, and inversion detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class TermStructureSignal(str, Enum):
    NORMAL = "NORMAL"          # front < back (upward sloping)
    INVERTED = "INVERTED"      # front > back (downward sloping)
    HUMPED = "HUMPED"          # peak in the middle
    FLAT = "FLAT"


@dataclass
class TermStructureSnapshot:
    tenors: np.ndarray   # shape (n,) in years
    atm_ivs: np.ndarray  # shape (n,) ATM implied vols

    @property
    def slope(self) -> float:
        """Linear slope of ATM vol vs tenor (OLS)."""
        if len(self.tenors) < 2:
            return 0.0
        coeffs = np.polyfit(self.tenors, self.atm_ivs, 1)
        return float(coeffs[0])

    @property
    def curvature(self) -> float:
        """Quadratic curvature (second-order fit coefficient)."""
        if len(self.tenors) < 3:
            return 0.0
        coeffs = np.polyfit(self.tenors, self.atm_ivs, 2)
        return float(coeffs[0])

    @property
    def front_to_back_spread(self) -> float:
        """ATM IV spread: longest tenor - shortest tenor."""
        return float(self.atm_ivs[-1] - self.atm_ivs[0])

    def classify(self, slope_threshold: float = 0.005, curv_threshold: float = 0.005) -> TermStructureSignal:
        slope = self.slope
        curv = self.curvature

        if abs(slope) < slope_threshold:
            return TermStructureSignal.FLAT
        # Humped: negative slope AND meaningfully positive curvature (local max in middle)
        if slope < 0 and curv > curv_threshold:
            return TermStructureSignal.HUMPED
        if slope < -slope_threshold:
            return TermStructureSignal.INVERTED
        return TermStructureSignal.NORMAL


def detect_inversion(
    tenors: np.ndarray,
    atm_ivs: np.ndarray,
    threshold: float = 0.01,
) -> list[tuple[float, float]]:
    """Return (T1, T2) pairs where atm_iv[T1] > atm_iv[T2] + threshold (T1 < T2)."""
    inversions = []
    for i in range(len(tenors) - 1):
        for j in range(i + 1, len(tenors)):
            if atm_ivs[i] > atm_ivs[j] + threshold:
                inversions.append((float(tenors[i]), float(tenors[j])))
    return inversions


def term_structure_zscore(
    current_slope: float,
    historical_slopes: np.ndarray,
) -> float:
    """Z-score of current term structure slope vs history."""
    if len(historical_slopes) < 2:
        return 0.0
    mu = float(np.mean(historical_slopes))
    sigma = float(np.std(historical_slopes, ddof=1))
    if sigma < 1e-10:
        return 0.0
    return (current_slope - mu) / sigma
