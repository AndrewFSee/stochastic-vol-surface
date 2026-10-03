"""Point-in-time implied-volatility features from one day's fitted surface.

Features are read off :class:`~src.surface.slices.ExpirySurface` — the
per-expiry SVI fits stored with each surface — not the 25x8 grid, so they are
available at any tenor (including 7 and 14 days) and at exact delta strikes.

Conventions
-----------
* Vols are annualised decimals (0.15 = 15%).
* Tenors are calendar days to expiry: ``7d`` means T = 7/365.
* Moneyness is forward log-moneyness; deltas are forward (undiscounted)
  Black deltas, so ``p25`` is the 25-delta put and ``c25`` the 25-delta call.
* **No extrapolation by default.**  A feature whose tenor lies outside the
  listed expiries, or whose strike lies outside the quotes of the bracketing
  expiries, is NaN rather than a model guess.  Missing is honest; a silently
  extrapolated value looks like data.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

TENORS_DAYS: dict[str, int] = {
    "7d": 7, "14d": 14, "30d": 30, "60d": 60, "91d": 91, "182d": 182, "365d": 365,
}
#: Tenors with full smile features (delta vols, risk reversal, butterfly).
SMILE_TENORS = ("30d", "91d")
#: Tenors with a model-free variance-swap vol.
VS_TENORS = ("30d", "91d")
DELTAS: dict[str, float] = {"p10": -0.10, "p25": -0.25, "c25": 0.25, "c10": 0.10}

#: Step for finite-difference smile derivatives at the money.
_DK = 0.01


def surface_features(
    slices: list[dict],
    spot: Optional[float] = None,
    *,
    allow_extrapolation: bool = False,
) -> dict[str, float]:
    """All implied-vol features for one surface.

    Parameters
    ----------
    slices
        Per-expiry fits, as stored in ``VolSurface.slices``.
    spot
        Underlying price at the snapshot; enables the implied carry feature.
    allow_extrapolation
        Return model values outside the quoted region instead of NaN.
    """
    from src.surface.slices import ExpirySurface

    nan = float("nan")
    f: dict[str, float] = {}
    if len(slices) < 2:
        return f
    surf = ExpirySurface(slices)

    def value(k: float, T: float) -> float:
        if not np.isfinite(k):
            return nan
        if not allow_extrapolation and not (surf.brackets(T) and bool(surf.is_observed(k, T))):
            return nan
        return float(surf.iv(k, T))

    # ── ATM term structure ────────────────────────────────────────────────
    for name, days in TENORS_DAYS.items():
        f[f"atm_{name}"] = value(0.0, days / 365)

    # ── Smile shape ──────────────────────────────────────────────────────
    for name in SMILE_TENORS:
        T = TENORS_DAYS[name] / 365
        atm = f[f"atm_{name}"]
        dv = {d: value(surf.delta_strike(delta, T), T) for d, delta in DELTAS.items()}
        for d, v in dv.items():
            f[f"iv_{d}_{name}"] = v
        f[f"rr25_{name}"] = dv["c25"] - dv["p25"]
        f[f"bf25_{name}"] = 0.5 * (dv["c25"] + dv["p25"]) - atm
        f[f"rr10_{name}"] = dv["c10"] - dv["p10"]
        f[f"bf10_{name}"] = 0.5 * (dv["c10"] + dv["p10"]) - atm

        # Slope and curvature of the smile in k at the money.
        up, dn = value(_DK, T), value(-_DK, T)
        f[f"atm_skew_{name}"] = (up - dn) / (2 * _DK)
        f[f"atm_curv_{name}"] = (up - 2 * atm + dn) / _DK ** 2

    # ── Model-free variance-swap vol ─────────────────────────────────────
    for name in VS_TENORS:
        f[f"vs_{name}"] = surf.variance_swap_vol(TENORS_DAYS[name] / 365)

    # ── Term structure ───────────────────────────────────────────────────
    f["ts_7_30"] = f["atm_30d"] - f["atm_7d"]
    f["ts_30_91"] = f["atm_91d"] - f["atm_30d"]
    f["ts_30_365"] = f["atm_365d"] - f["atm_30d"]

    # ── Implied carry ────────────────────────────────────────────────────
    # ln(F/S)/T on the parity-forward expiry nearest one year: the market's
    # rate minus dividend yield minus borrow, as one number.  Its level is
    # informative but its daily change is mostly noise (~0.1%/yr, lag-1
    # autocorr ~ -0.4): the snapshot spot and the option quotes are not
    # perfectly synchronous.  Smooth it before use.  (A spot-free
    # forward-to-forward version was tried; it halves SPY's noise but is worse
    # for single stocks, whose quarterly dividends land between the two
    # expiries.)
    f["fwd_carry_1y"] = nan
    if spot and spot > 0:
        parity = [s for s in surf.slices if s.forward_source == "parity" and 0.5 <= s.T <= 1.5]
        if parity:
            s = min(parity, key=lambda s: abs(s.T - 1.0))
            f["fwd_carry_1y"] = math.log(s.forward / spot) / s.T

    # ── Quality ──────────────────────────────────────────────────────────
    f["n_expiries"] = float(len(surf.slices))
    f["nearest_expiry_days"] = float(surf.t_min * 365)
    f["fit_rmse"] = float(np.median([s.rmse_iv for s in surf.slices]))
    f["parity_fraction"] = float(np.mean([s.forward_source == "parity" for s in surf.slices]))
    return f
