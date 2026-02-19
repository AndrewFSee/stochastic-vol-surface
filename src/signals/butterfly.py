"""
Butterfly spread mispricing detection (model vs market).

A butterfly spread (buy 1 ITM, sell 2 ATM, buy 1 OTM) should be non-negative
in the absence of arbitrage.  When the market butterfly price diverges
significantly from the model price, it signals a potential trade.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def butterfly_spread_iv(
    iv_low: float,
    iv_atm: float,
    iv_high: float,
    k_low: float,
    k_atm: float,
    k_high: float,
) -> float:
    """Compute the vol-space butterfly spread value.

    Approximates the butterfly as:
        BF = (σ_low + σ_high) / 2 - σ_atm
    This is the raw 'butterfly vol' rather than a price.

    Positive BF → smile is convex (normal).
    Negative BF → smile concavity (potential butterfly arb).

    Parameters
    ----------
    iv_low, iv_atm, iv_high:
        Implied vols at the three strikes.
    k_low, k_atm, k_high:
        Corresponding log-moneyness values.

    Returns
    -------
    float
        Butterfly implied vol spread.
    """
    # Normalise to unit spacing
    dk = (k_high - k_low) / 2.0
    if dk <= 0:
        return float("nan")
    return float((iv_low + iv_high) / 2.0 - iv_atm)


def detect_butterfly_mispricing(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    tenor_years: float,
    k_spread: float = 0.05,
    threshold: float = -0.005,
) -> dict[str, float]:
    """Detect butterfly mispricing at a specific tenor.

    Tests whether the smile curvature at ATM (k=0) is consistent with
    positive local variance density.

    Parameters
    ----------
    surface:
        VolSurface object.
    tenor_years:
        Tenor to check.
    k_spread:
        Half-width of the butterfly in log-moneyness (default 5%).
    threshold:
        Signal threshold: flag if butterfly spread < threshold.

    Returns
    -------
    dict
        ``{bf_spread, threshold, signal}``
        where signal = 1 (buy butterfly) or 0 (no signal).
    """
    iv_low = surface.get_iv(-k_spread, tenor_years)
    iv_atm = surface.get_iv(0.0, tenor_years)
    iv_high = surface.get_iv(k_spread, tenor_years)

    bf = butterfly_spread_iv(iv_low, iv_atm, iv_high, -k_spread, 0.0, k_spread)
    signal = 1 if bf < threshold else 0

    return {"bf_spread": bf, "threshold": threshold, "signal": signal}


def butterfly_signal_surface(
    surface: "VolSurface",  # type: ignore[name-defined]  # noqa: F821
    tenors: list[float] | None = None,
    k_spread: float = 0.05,
    threshold: float = -0.005,
) -> pd.DataFrame:
    """Compute butterfly mispricing signals across all configured tenors.

    Returns
    -------
    pd.DataFrame
        One row per tenor with columns ``bf_spread``, ``signal``.
    """
    if tenors is None:
        tenors = [1 / 52, 1 / 12, 3 / 12, 6 / 12, 1.0]

    rows = []
    for T in tenors:
        result = detect_butterfly_mispricing(surface, T, k_spread, threshold)
        rows.append({"tenor_years": T, **result})

    return pd.DataFrame(rows).set_index("tenor_years")
