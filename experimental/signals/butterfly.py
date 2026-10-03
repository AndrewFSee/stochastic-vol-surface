"""Butterfly spread mispricing: model vs market."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ButterflyMispricing:
    strike_lo: float
    strike_mid: float
    strike_hi: float
    market_butterfly_iv: float   # (IV_lo + IV_hi)/2 - IV_mid
    model_butterfly_iv: float
    mispricing_bps: float        # (market - model) * 10000


def compute_butterfly_mispricing(
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    model_ivs: np.ndarray,
    threshold_bps: float = 50.0,
) -> list[ButterflyMispricing]:
    """Identify butterfly mispricings between model and market IVs.

    For each interior strike, constructs a symmetric butterfly and
    compares model vs market IV.

    Parameters
    ----------
    strikes     : sorted array of strike prices
    market_ivs  : market implied vols (same length)
    model_ivs   : model-fitted implied vols (same length)
    threshold_bps : minimum mispricing (in bps of IV) to flag

    Returns
    -------
    list of ButterflyMispricing for strikes exceeding the threshold
    """
    n = len(strikes)
    results = []

    for i in range(1, n - 1):
        # symmetric butterfly around strikes[i]
        mkt_bf = (market_ivs[i - 1] + market_ivs[i + 1]) / 2 - market_ivs[i]
        mod_bf = (model_ivs[i - 1] + model_ivs[i + 1]) / 2 - model_ivs[i]
        mispricing = (mkt_bf - mod_bf) * 10_000  # convert to bps

        if abs(mispricing) >= threshold_bps:
            results.append(
                ButterflyMispricing(
                    strike_lo=float(strikes[i - 1]),
                    strike_mid=float(strikes[i]),
                    strike_hi=float(strikes[i + 1]),
                    market_butterfly_iv=float(mkt_bf),
                    model_butterfly_iv=float(mod_bf),
                    mispricing_bps=float(mispricing),
                )
            )
    return results


def butterfly_iv(
    iv_lo: float,
    iv_mid: float,
    iv_hi: float,
) -> float:
    """25-delta butterfly: (OTM put IV + OTM call IV) / 2 - ATM IV."""
    return (iv_lo + iv_hi) / 2 - iv_mid
