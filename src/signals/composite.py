"""Aggregate signals into tradeable vol recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.signals.skew_signals import SkewSignal, compute_skew_signal
from src.signals.term_structure import TermStructureSignal, TermStructureSnapshot
from src.signals.regime_vol import VolRegime, classify_regime


class TradeRecommendation(str, Enum):
    BUY_VOL = "BUY_VOL"        # buy straddles / vega
    SELL_VOL = "SELL_VOL"      # sell straddles / collect premium
    BUY_SKEW = "BUY_SKEW"      # buy risk reversal (call vs put)
    SELL_SKEW = "SELL_SKEW"    # sell risk reversal
    STEEPEN_TS = "STEEPEN_TS"  # buy front / sell back (calendar)
    FLATTEN_TS = "FLATTEN_TS"  # sell front / buy back (calendar)
    NEUTRAL = "NEUTRAL"


@dataclass
class CompositeSignal:
    skew_signal: SkewSignal = SkewSignal.NEUTRAL
    term_structure_signal: TermStructureSignal = TermStructureSignal.NORMAL
    vol_regime: VolRegime = VolRegime.NORMAL
    atm_vol_zscore: float = 0.0
    recommendation: TradeRecommendation = TradeRecommendation.NEUTRAL
    confidence: float = 0.0
    rationale: str = ""


def aggregate_signals(
    current_skew: float,
    historical_skews: np.ndarray,
    ts_snapshot: TermStructureSnapshot,
    current_vix: float,
    current_atm_vol: float,
    historical_atm_vols: np.ndarray,
    skew_z_threshold: float = 1.5,
    vol_z_threshold: float = 1.5,
) -> CompositeSignal:
    """Combine individual signals into a single composite recommendation."""
    # Skew signal
    skew_sig = compute_skew_signal(current_skew, historical_skews, skew_z_threshold)

    # Term structure signal
    ts_sig = ts_snapshot.classify()

    # Regime
    regime = classify_regime(current_vix)

    # ATM vol z-score
    if len(historical_atm_vols) >= 2:
        mu = float(np.mean(historical_atm_vols))
        std = float(np.std(historical_atm_vols, ddof=1))
        atm_z = (current_atm_vol - mu) / max(std, 1e-10)
    else:
        atm_z = 0.0

    # Determine recommendation
    scores: dict[TradeRecommendation, float] = {r: 0.0 for r in TradeRecommendation}

    # Vol level signal
    if atm_z > vol_z_threshold:
        scores[TradeRecommendation.SELL_VOL] += abs(atm_z)
    elif atm_z < -vol_z_threshold:
        scores[TradeRecommendation.BUY_VOL] += abs(atm_z)

    # Skew signal
    if skew_sig == SkewSignal.STEEPENING:
        scores[TradeRecommendation.BUY_SKEW] += 1.0
    elif skew_sig == SkewSignal.FLATTENING:
        scores[TradeRecommendation.SELL_SKEW] += 1.0

    # Term structure signal
    if ts_sig == TermStructureSignal.INVERTED:
        scores[TradeRecommendation.FLATTEN_TS] += 1.0
    elif ts_sig == TermStructureSignal.NORMAL:
        scores[TradeRecommendation.STEEPEN_TS] += 0.5

    # Crisis regime: prefer buying vol
    if regime == VolRegime.CRISIS:
        scores[TradeRecommendation.BUY_VOL] += 2.0
    elif regime == VolRegime.LOW:
        scores[TradeRecommendation.SELL_VOL] += 1.0

    best_rec = max(scores, key=lambda r: scores[r])
    confidence = scores[best_rec] / max(sum(scores.values()), 1e-10)

    if scores[best_rec] < 0.1:
        best_rec = TradeRecommendation.NEUTRAL
        confidence = 0.0

    rationale = (
        f"Skew={skew_sig.value}, TS={ts_sig.value}, "
        f"Regime={regime.value}, ATM-z={atm_z:.2f}"
    )

    return CompositeSignal(
        skew_signal=skew_sig,
        term_structure_signal=ts_sig,
        vol_regime=regime,
        atm_vol_zscore=atm_z,
        recommendation=best_rec,
        confidence=confidence,
        rationale=rationale,
    )
