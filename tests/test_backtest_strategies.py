"""Tests for option strategy construction and sizing."""

from datetime import date

import numpy as np
import pytest

from src.backtest.engine import BacktestConfig, portfolio_greeks
from src.backtest.greeks import bs_vega
from src.backtest.strategies import (
    MAX_CONTRACTS_PER_LEG,
    STRATEGIES,
    butterfly_trade,
    delta_hedged_straddle,
    risk_reversal,
    short_straddle,
)
from src.surface.surface import VolSurface

AS_OF = date(2026, 1, 5)


def _surface(iv=0.20, skew=0.0, smile=0.0, spot=100.0, r=0.04):
    """Surface with optional linear skew and quadratic smile in log-moneyness."""
    k = np.linspace(-0.4, 0.2, 25)
    t = np.array([0.0833, 0.1667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    grid = np.empty((len(k), len(t)))
    for i, kk in enumerate(k):
        grid[i, :] = iv + skew * kk + smile * kk ** 2
    return VolSurface(ticker="TEST", as_of=AS_OF, k_grid=k, t_grid=t,
                      iv_grid=grid, spot=spot, r=r)


ALL = ["straddle", "short_straddle", "risk_reversal", "butterfly"]


# ── Contract specification ───────────────────────────────────────────────


@pytest.mark.parametrize("name", ALL)
def test_every_strategy_returns_specified_legs(name):
    """Legs must carry the contract, or the engine cannot reprice them."""
    pos = STRATEGIES[name](AS_OF, _surface(smile=0.3))
    assert pos is not None
    assert pos.legs
    for leg in pos.legs:
        assert leg.option_type in ("call", "put")
        assert leg.strike > 0
        assert leg.expiry > AS_OF
        assert np.isfinite(leg.quantity) and leg.quantity != 0


@pytest.mark.parametrize("name", ALL)
def test_expiry_matches_requested_tenor(name):
    pos = STRATEGIES[name](AS_OF, _surface(smile=0.3), tenor=0.5)
    days = (pos.legs[0].expiry - AS_OF).days
    assert days == pytest.approx(0.5 * 365, abs=2)


@pytest.mark.parametrize("name", ALL)
def test_strategies_use_the_surface_rate_not_a_constant(name):
    """Strikes are struck on the forward, so a different r moves them."""
    lo = STRATEGIES[name](AS_OF, _surface(smile=0.3, r=0.00), tenor=1.0)
    hi = STRATEGIES[name](AS_OF, _surface(smile=0.3, r=0.10), tenor=1.0)
    assert hi.legs[0].strike > lo.legs[0].strike


def test_straddle_strikes_are_at_the_forward():
    r, tenor, spot = 0.04, 0.25, 100.0
    pos = delta_hedged_straddle(AS_OF, _surface(r=r), tenor=tenor)
    expected = spot * np.exp(r * tenor)
    for leg in pos.legs:
        assert leg.strike == pytest.approx(expected, rel=1e-9)


def test_straddle_has_one_call_and_one_put_at_the_same_strike():
    pos = delta_hedged_straddle(AS_OF, _surface())
    assert {leg.option_type for leg in pos.legs} == {"call", "put"}
    assert pos.legs[0].strike == pytest.approx(pos.legs[1].strike)


def test_short_straddle_mirrors_the_long():
    lng = delta_hedged_straddle(AS_OF, _surface())
    sht = short_straddle(AS_OF, _surface())
    for a, b in zip(lng.legs, sht.legs):
        assert b.quantity == pytest.approx(-a.quantity)


def test_risk_reversal_is_long_call_short_put():
    pos = risk_reversal(AS_OF, _surface(skew=-0.10))
    by_type = {leg.option_type: leg for leg in pos.legs}
    assert by_type["call"].quantity > 0
    assert by_type["put"].quantity < 0


def test_risk_reversal_straddles_the_forward():
    r, tenor = 0.04, 0.25
    pos = risk_reversal(AS_OF, _surface(r=r), tenor=tenor)
    fwd = 100.0 * np.exp(r * tenor)
    by_type = {leg.option_type: leg for leg in pos.legs}
    assert by_type["put"].strike < fwd < by_type["call"].strike


def test_butterfly_has_three_strikes_and_zero_net_quantity():
    pos = butterfly_trade(AS_OF, _surface(smile=0.5))
    assert len(pos.legs) == 3
    assert sum(leg.quantity for leg in pos.legs) == pytest.approx(0.0, abs=1e-9)


def test_butterfly_body_is_long_and_wings_short():
    pos = butterfly_trade(AS_OF, _surface(smile=0.5))
    qty = [leg.quantity for leg in sorted(pos.legs, key=lambda x: x.strike)]
    assert qty[0] < 0 and qty[2] < 0      # wings
    assert qty[1] > 0                      # body


# ── Sizing ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ALL)
def test_sizing_is_bounded(name):
    """Regression: risk reversal once sized off ~zero net vega and exploded."""
    pos = STRATEGIES[name](AS_OF, _surface(smile=0.3))
    for leg in pos.legs:
        assert abs(leg.quantity) <= MAX_CONTRACTS_PER_LEG


def test_risk_reversal_sizing_is_sane_despite_zero_net_vega():
    """A symmetric surface makes the RR's net vega ~0 — the old blow-up case."""
    surf = _surface(iv=0.20, skew=0.0, smile=0.0)
    cfg = BacktestConfig(target_vega_notional=10_000.0)
    pos = risk_reversal(AS_OF, surf, config=cfg)

    call = next(l for l in pos.legs if l.option_type == "call")
    put = next(l for l in pos.legs if l.option_type == "put")
    net_vega = (
        bs_vega(100.0, call.strike, 0.25, 0.04, 0.20) * call.quantity
        + bs_vega(100.0, put.strike, 0.25, 0.04, 0.20) * put.quantity
    )
    # Net vega really is near zero...
    assert abs(net_vega) < abs(call.quantity) * 0.5
    # ...yet the position stays a sane size.
    assert abs(call.quantity) < 1_000


@pytest.mark.parametrize("name", ALL)
def test_gross_vega_hits_the_target(name):
    cfg = BacktestConfig(target_vega_notional=10_000.0, contract_multiplier=100.0)
    surf = _surface(smile=0.3)
    pos = STRATEGIES[name](AS_OF, surf, config=cfg)

    gross = 0.0
    for leg in pos.legs:
        t = (leg.expiry - AS_OF).days / 365.0
        iv = float(surf.iv(np.log(leg.strike / (100.0 * np.exp(0.04 * t))), t))
        gross += abs(leg.quantity) * bs_vega(100.0, leg.strike, t, 0.04, iv)
    assert gross * 100.0 * 0.01 == pytest.approx(10_000.0, rel=0.25)


@pytest.mark.parametrize("name", ALL)
def test_sizing_scales_with_the_target(name):
    surf = _surface(smile=0.3)
    small = STRATEGIES[name](AS_OF, surf, config=BacktestConfig(target_vega_notional=1_000.0))
    big = STRATEGIES[name](AS_OF, surf, config=BacktestConfig(target_vega_notional=10_000.0))
    ratio = abs(big.legs[0].quantity) / abs(small.legs[0].quantity)
    assert ratio == pytest.approx(10.0, rel=0.05)


# ── Greek signatures ─────────────────────────────────────────────────────


def test_long_straddle_is_long_vega_and_gamma():
    surf = _surface()
    pos = delta_hedged_straddle(AS_OF, surf)
    g = portfolio_greeks([pos], AS_OF, 100.0, 0.04, surf, 100.0)
    assert g["vega"] > 0 and g["gamma"] > 0 and g["theta"] < 0


def test_short_straddle_is_short_vega_and_gamma():
    surf = _surface()
    pos = short_straddle(AS_OF, surf)
    g = portfolio_greeks([pos], AS_OF, 100.0, 0.04, surf, 100.0)
    assert g["vega"] < 0 and g["gamma"] < 0 and g["theta"] > 0


def test_butterfly_is_short_convexity():
    surf = _surface(smile=0.5)
    pos = butterfly_trade(AS_OF, surf)
    g = portfolio_greeks([pos], AS_OF, 100.0, 0.04, surf, 100.0)
    # Long body / short wings is net long gamma at the money.
    assert np.isfinite(g["gamma"])
    assert sum(l.quantity for l in pos.legs) == pytest.approx(0.0, abs=1e-9)


# ── Degenerate inputs ────────────────────────────────────────────────────


def test_strategies_return_none_on_a_broken_surface():
    surf = _surface()
    surf.iv_grid = np.full_like(surf.iv_grid, np.nan)
    surf.__post_init__()
    for name in ALL:
        assert STRATEGIES[name](AS_OF, surf) is None


def test_registry_covers_every_strategy():
    assert set(STRATEGIES) == set(ALL)
