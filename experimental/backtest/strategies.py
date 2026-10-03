"""Option strategies: delta-hedged straddles, risk reversals, butterflies.

Each strategy returns a :class:`~experimental.backtest.engine.Position` whose legs carry
full contract specifications (type, strike, expiry, signed quantity), so the
engine can reprice them against any later surface.

Sizing
------
All strategies size to ``config.target_vega_notional``, the cash P&L per one
vol point (0.01) of parallel IV move.  This makes strategies comparable: a
straddle and a butterfly with the same target start with the same vol
exposure, so differences in result come from their shape, not their size.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from experimental.backtest.engine import BacktestConfig, OptionLeg, Position
from experimental.backtest.greeks import bs_vega

if TYPE_CHECKING:
    from src.surface.surface import VolSurface

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0
VOL_POINT = 0.01


def _expiry_for_tenor(as_of: date, tenor: float) -> date:
    """Calendar expiry date for a tenor expressed in years."""
    return as_of + timedelta(days=int(round(tenor * DAYS_PER_YEAR)))


def _strike_from_logmoneyness(spot: float, r: float, tenor: float, k: float) -> float:
    """Invert ``k = ln(K/F)`` with ``F = S·e^{rT}`` to get the strike."""
    return float(spot * np.exp(r * tenor) * np.exp(k))


def _surface_iv(surface: "VolSurface", k: float, tenor: float, fallback: float) -> float:
    """Query the surface, falling back if it returns something unusable."""
    try:
        iv = float(surface.iv(k, tenor))
    except Exception:
        return fallback
    return iv if np.isfinite(iv) and iv > 0 else fallback


#: Cap on contracts per leg.  Guards against a degenerate sizing division
#: silently producing an absurdly large position.
MAX_CONTRACTS_PER_LEG = 100_000.0


def _size_to_vega(
    legs_spec: list[tuple[str, float, float]],   # (option_type, strike, iv)
    spot: float, tenor: float, r: float,
    target_vega_notional: float, multiplier: float,
    weights: list[float],
) -> float:
    """Scale factor so the position's *gross* vega equals the target.

    Gross (sum of absolute leg vegas), not net: a risk reversal is long one
    wing and short the other, so its net vega is structurally near zero.
    Sizing off net vega divides by ~0 and produces an unbounded position —
    which is exactly what it did before this was fixed.  Gross vega is
    well-defined for every structure here and keeps their absolute vol
    exposure comparable.

    ``weights`` are the relative signed quantities of each leg.
    """
    gross_vega = 0.0
    for (_opt_type, strike, iv), w in zip(legs_spec, weights):
        gross_vega += abs(w) * bs_vega(spot, strike, tenor, r, iv)

    # Vega is per unit vol; convert the target (cash per vol point) to match.
    denom = gross_vega * multiplier * VOL_POINT
    if denom < 1e-9:
        return 0.0

    scale = float(target_vega_notional / denom)
    max_weight = max(abs(w) for w in weights) or 1.0
    if scale * max_weight > MAX_CONTRACTS_PER_LEG:
        logger.warning("Sizing capped: %.0f contracts/leg requested", scale * max_weight)
        scale = MAX_CONTRACTS_PER_LEG / max_weight
    return scale


# ────────────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────────────


def delta_hedged_straddle(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    config: Optional[BacktestConfig] = None,
) -> Optional[Position]:
    """Long an ATM straddle (call + put at the forward).

    Delta hedging is applied by the engine, so this is a bet on realised
    vol exceeding implied: it earns gamma when the underlying moves and pays
    theta when it does not.
    """
    config = config or BacktestConfig()
    spot = float(surface.spot)
    r = float(getattr(surface, "r", 0.05))
    atm_iv = float(surface.atm_vol(tenor))
    if not np.isfinite(atm_iv) or atm_iv <= 0:
        return None

    strike = _strike_from_logmoneyness(spot, r, tenor, 0.0)
    expiry = _expiry_for_tenor(dt, tenor)

    spec = [("call", strike, atm_iv), ("put", strike, atm_iv)]
    weights = [1.0, 1.0]
    scale = _size_to_vega(spec, spot, tenor, r, config.target_vega_notional,
                          config.contract_multiplier, weights)
    if scale <= 0:
        return None

    return Position(
        open_date=dt, ticker=ticker, strategy="delta_hedged_straddle",
        legs=[
            OptionLeg("call", strike, expiry, weights[0] * scale),
            OptionLeg("put", strike, expiry, weights[1] * scale),
        ],
    )


def risk_reversal(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    delta: float = 0.25,
    config: Optional[BacktestConfig] = None,
) -> Optional[Position]:
    """25-delta risk reversal: long the call wing, short the put wing.

    A direct bet that the put skew is too expensive relative to the call.
    """
    config = config or BacktestConfig()
    spot = float(surface.spot)
    r = float(getattr(surface, "r", 0.05))
    atm_iv = float(surface.atm_vol(tenor))
    if not np.isfinite(atm_iv) or atm_iv <= 0:
        return None

    # Standard approximation of the 25-delta strikes in log-moneyness.
    dk = delta * atm_iv * np.sqrt(tenor)
    call_k, put_k = dk, -dk
    call_iv = _surface_iv(surface, call_k, tenor, atm_iv)
    put_iv = _surface_iv(surface, put_k, tenor, atm_iv)

    call_strike = _strike_from_logmoneyness(spot, r, tenor, call_k)
    put_strike = _strike_from_logmoneyness(spot, r, tenor, put_k)
    expiry = _expiry_for_tenor(dt, tenor)

    spec = [("call", call_strike, call_iv), ("put", put_strike, put_iv)]
    weights = [1.0, -1.0]
    scale = _size_to_vega(spec, spot, tenor, r, config.target_vega_notional,
                          config.contract_multiplier, weights)
    if scale <= 0:
        return None

    return Position(
        open_date=dt, ticker=ticker, strategy="risk_reversal",
        legs=[
            OptionLeg("call", call_strike, expiry, weights[0] * scale),
            OptionLeg("put", put_strike, expiry, weights[1] * scale),
        ],
    )


def butterfly_trade(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    wing_width: float = 0.05,
    config: Optional[BacktestConfig] = None,
) -> Optional[Position]:
    """Short ATM butterfly: sell the wings, buy the body.

    Long the body / short the wings is a bet that the smile's convexity is
    overpriced — it profits when the surface flattens.
    """
    config = config or BacktestConfig()
    spot = float(surface.spot)
    r = float(getattr(surface, "r", 0.05))
    atm_iv = float(surface.atm_vol(tenor))
    if not np.isfinite(atm_iv) or atm_iv <= 0:
        return None

    lo_iv = _surface_iv(surface, -wing_width, tenor, atm_iv)
    hi_iv = _surface_iv(surface, wing_width, tenor, atm_iv)

    lo_k = _strike_from_logmoneyness(spot, r, tenor, -wing_width)
    mid_k = _strike_from_logmoneyness(spot, r, tenor, 0.0)
    hi_k = _strike_from_logmoneyness(spot, r, tenor, wing_width)
    expiry = _expiry_for_tenor(dt, tenor)

    # Long body, short wings — net short convexity.
    spec = [("call", lo_k, lo_iv), ("call", mid_k, atm_iv), ("call", hi_k, hi_iv)]
    weights = [-1.0, 2.0, -1.0]
    scale = _size_to_vega(spec, spot, tenor, r, config.target_vega_notional,
                          config.contract_multiplier, weights)
    if scale <= 0:
        return None

    return Position(
        open_date=dt, ticker=ticker, strategy="butterfly",
        legs=[
            OptionLeg("call", lo_k, expiry, weights[0] * scale),
            OptionLeg("call", mid_k, expiry, weights[1] * scale),
            OptionLeg("call", hi_k, expiry, weights[2] * scale),
        ],
    )


def short_straddle(
    dt: date,
    surface: "VolSurface",
    ticker: str = "SPY",
    tenor: float = 0.25,
    config: Optional[BacktestConfig] = None,
) -> Optional[Position]:
    """Short ATM straddle — the classic short-vol / vol-risk-premium harvest."""
    pos = delta_hedged_straddle(dt, surface, ticker=ticker, tenor=tenor, config=config)
    if pos is None:
        return None
    for leg in pos.legs:
        leg.quantity = -leg.quantity
    pos.strategy = "short_straddle"
    return pos


STRATEGIES = {
    "straddle": delta_hedged_straddle,
    "short_straddle": short_straddle,
    "risk_reversal": risk_reversal,
    "butterfly": butterfly_trade,
}
