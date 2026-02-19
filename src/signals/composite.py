"""
Composite signal aggregation: combine all vol-arb signals into tradeable
recommendations with confidence scores.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_signals(
    skew_signal: int,
    ts_signal: int,
    bf_signal: int,
    regime: str,
    skew_zscore: Optional[float] = None,
    ts_zscore: Optional[float] = None,
) -> dict:
    """Aggregate individual vol signals into a composite recommendation.

    Parameters
    ----------
    skew_signal:
        From skew_signals: +1 (steep), -1 (flat), 0 (neutral).
    ts_signal:
        From term_structure: +1 (sell front/buy back), -1 (buy front), 0.
    bf_signal:
        From butterfly: +1 (buy butterfly), 0 (neutral).
    regime:
        Current vol regime label.
    skew_zscore, ts_zscore:
        Z-scores for confidence weighting.

    Returns
    -------
    dict
        ``{composite_score, action, confidence, regime}``.
    """
    # Composite score: average of non-zero signals
    signals = [s for s in [skew_signal, ts_signal, bf_signal] if s != 0]
    composite = float(sum(signals) / len(signals)) if signals else 0.0

    # Reduce position sizing in high-vol regimes
    regime_scale = {"Low": 1.0, "Normal": 1.0, "High": 0.5, "Crisis": 0.25}.get(regime, 1.0)
    composite_scaled = composite * regime_scale

    # Action: long vol (+), short vol (-), or flat
    if composite_scaled > 0.3:
        action = "buy_vol"
    elif composite_scaled < -0.3:
        action = "sell_vol"
    else:
        action = "flat"

    # Confidence from Z-score magnitude
    z_scores = [abs(z) for z in [skew_zscore, ts_zscore] if z is not None]
    confidence = float(sum(z_scores) / len(z_scores)) if z_scores else 0.0

    return {
        "composite_score": composite_scaled,
        "action": action,
        "confidence": min(confidence, 5.0),
        "regime": regime,
    }
