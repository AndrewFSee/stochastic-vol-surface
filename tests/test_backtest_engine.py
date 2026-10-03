"""Tests for the mark-to-market backtest engine."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    OptionLeg,
    Position,
    leg_iv,
    portfolio_greeks,
    price_leg,
    run_backtest,
)
from src.backtest.greeks import bs_price
from src.surface.surface import VolSurface


# ── Fixtures ─────────────────────────────────────────────────────────────


def _flat_surface(as_of: date, spot: float = 100.0, iv: float = 0.20, r: float = 0.04):
    """A surface that is flat in both moneyness and tenor."""
    k = np.linspace(-0.4, 0.2, 25)
    t = np.array([0.0833, 0.1667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    return VolSurface(
        ticker="TEST", as_of=as_of, k_grid=k, t_grid=t,
        iv_grid=np.full((len(k), len(t)), iv), spot=spot, r=r,
    )


def _history(n=40, spot0=100.0, drift=0.0, shock=None, iv=0.20):
    """Build a dated surface history with a controllable spot path."""
    out = {}
    d = date(2026, 1, 5)
    for i in range(n):
        spot = spot0 * (1 + drift) ** i
        if shock is not None and i >= n // 2:
            spot *= shock
        out[d] = _flat_surface(d, spot=spot, iv=iv)
        d += timedelta(days=1)
    return out


def _straddle(dt, surface, ticker="TEST", config=None, tenor=0.25, qty=1.0):
    """A fixed-size ATM straddle, independent of the sizing logic."""
    strike = float(surface.spot)
    expiry = dt + timedelta(days=int(tenor * 365))
    return Position(
        open_date=dt, ticker=ticker, strategy="test_straddle",
        legs=[OptionLeg("call", strike, expiry, qty),
              OptionLeg("put", strike, expiry, qty)],
    )


def _flat(dt, surface, ticker="TEST", config=None):
    """A strategy that never trades."""
    return None


# ── Pricing primitives ───────────────────────────────────────────────────


def test_bs_price_put_call_parity():
    S, K, T, r, sig = 100.0, 95.0, 0.5, 0.04, 0.2
    c = bs_price(S, K, T, r, sig, "call")
    p = bs_price(S, K, T, r, sig, "put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


def test_bs_price_at_expiry_is_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, 0.04, 0.2, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, 0.04, 0.2, "put") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, 0.04, 0.2, "call") == pytest.approx(0.0)


def test_bs_price_increases_with_vol():
    args = (100.0, 100.0, 0.5, 0.04)
    assert bs_price(*args, 0.30, "call") > bs_price(*args, 0.15, "call")


def test_leg_years_to_expiry_floors_at_zero():
    leg = OptionLeg("call", 100.0, date(2026, 1, 1), 1.0)
    assert leg.years_to_expiry(date(2026, 6, 1)) == 0.0


# ── Surface lookup for a leg ─────────────────────────────────────────────


def test_leg_iv_reads_the_surface():
    d = date(2026, 1, 5)
    surf = _flat_surface(d, iv=0.25)
    leg = OptionLeg("call", 100.0, d + timedelta(days=90), 1.0)
    assert leg_iv(surf, leg, d, 100.0, 0.04) == pytest.approx(0.25, abs=1e-6)


def test_leg_iv_falls_back_when_expired():
    d = date(2026, 1, 5)
    surf = _flat_surface(d)
    leg = OptionLeg("call", 100.0, d, 1.0)
    leg.last_iv = 0.22
    assert leg_iv(surf, leg, d, 100.0, 0.04) == pytest.approx(0.22)


def test_price_leg_matches_black_scholes():
    d = date(2026, 1, 5)
    leg = OptionLeg("call", 100.0, d + timedelta(days=365), 1.0)
    got = price_leg(leg, d, 100.0, 0.04, 0.20)
    assert got == pytest.approx(bs_price(100.0, 100.0, 365 / 365.0, 0.04, 0.20, "call"))


# ── Portfolio Greeks ─────────────────────────────────────────────────────


def test_straddle_is_delta_neutral_at_the_money():
    d = date(2026, 1, 5)
    surf = _flat_surface(d)
    pos = _straddle(d, surf)
    g = portfolio_greeks([pos], d, 100.0, 0.04, surf, 100.0)
    # ATM call delta ~ +0.5, put ~ -0.5 (plus a small forward effect).
    assert abs(g["delta"]) < 15.0
    assert g["vega"] > 0
    assert g["gamma"] > 0


def test_long_options_have_negative_theta():
    d = date(2026, 1, 5)
    surf = _flat_surface(d)
    g = portfolio_greeks([_straddle(d, surf)], d, 100.0, 0.04, surf, 100.0)
    assert g["theta"] < 0


def test_short_position_flips_greek_signs():
    d = date(2026, 1, 5)
    surf = _flat_surface(d)
    lng = portfolio_greeks([_straddle(d, surf, qty=1.0)], d, 100.0, 0.04, surf, 100.0)
    sht = portfolio_greeks([_straddle(d, surf, qty=-1.0)], d, 100.0, 0.04, surf, 100.0)
    assert sht["vega"] == pytest.approx(-lng["vega"])
    assert sht["theta"] == pytest.approx(-lng["theta"])


def test_hedge_shares_enter_portfolio_delta():
    d = date(2026, 1, 5)
    surf = _flat_surface(d)
    pos = _straddle(d, surf)
    base = portfolio_greeks([pos], d, 100.0, 0.04, surf, 100.0)["delta"]
    pos.hedge_shares = -50.0
    after = portfolio_greeks([pos], d, 100.0, 0.04, surf, 100.0)["delta"]
    assert after == pytest.approx(base - 50.0)


# ── Engine accounting ────────────────────────────────────────────────────


def test_empty_history_returns_empty_frame():
    assert run_backtest({}, strategy_fn=_straddle).empty


def test_produces_one_row_per_date():
    hist = _history(20)
    out = run_backtest(hist, strategy_fn=_straddle)
    assert len(out) == 20


def test_has_columns_the_tearsheet_needs():
    out = run_backtest(_history(20), strategy_fn=_straddle)
    for col in ("date", "gross_pnl", "net_pnl", "portfolio_value", "cumulative_pnl"):
        assert col in out.columns


def test_net_equals_gross_minus_costs():
    out = run_backtest(_history(30), strategy_fn=_straddle)
    np.testing.assert_allclose(
        out.net_pnl, out.gross_pnl - out.transaction_costs, atol=1e-9,
    )


def test_portfolio_value_tracks_cumulative_net_pnl():
    cfg = BacktestConfig(initial_capital=1_000_000.0)
    out = run_backtest(_history(30), strategy_fn=_straddle, config=cfg)
    assert out.portfolio_value.iloc[-1] == pytest.approx(
        1_000_000.0 + out.net_pnl.sum(), rel=1e-9,
    )


def test_attribution_sums_to_gross_pnl():
    """The Greek decomposition plus the residual must reconstruct P&L exactly."""
    out = run_backtest(_history(40, drift=0.002), strategy_fn=_straddle)
    total = out[["delta_pnl", "gamma_pnl", "vega_pnl",
                 "theta_pnl", "hedge_pnl", "unexplained_pnl"]].sum().sum()
    assert total == pytest.approx(out.gross_pnl.sum(), rel=1e-9)


def test_date_range_filter():
    hist = _history(30)
    dates = sorted(hist)
    out = run_backtest(hist, strategy_fn=_straddle,
                       start_date=dates[5], end_date=dates[14])
    assert len(out) == 10
    assert out.date.iloc[0] == dates[5]


def test_flat_strategy_produces_no_pnl():
    out = run_backtest(_history(20), strategy_fn=_flat)
    assert out.net_pnl.abs().sum() == pytest.approx(0.0)
    assert (out.n_positions == 0).all()


def test_strategy_exception_is_contained():
    def boom(dt, surface, ticker="TEST", config=None):
        raise RuntimeError("strategy blew up")

    out = run_backtest(_history(10), strategy_fn=boom)
    assert len(out) == 10
    assert (out.n_positions == 0).all()


# ── Hedging behaviour ────────────────────────────────────────────────────


def test_delta_hedging_neutralises_delta():
    cfg = BacktestConfig(delta_hedge=True)
    out = run_backtest(_history(30, drift=0.003), strategy_fn=_straddle, config=cfg)
    assert out.net_delta.abs().max() < 1e-6


def test_without_hedging_delta_is_left_open():
    cfg = BacktestConfig(delta_hedge=False)
    out = run_backtest(_history(30, drift=0.01), strategy_fn=_straddle, config=cfg)
    assert out.net_delta.abs().max() > 1e-6


def test_hedge_pnl_offsets_delta_pnl_when_hedged():
    cfg = BacktestConfig(delta_hedge=True)
    out = run_backtest(_history(30, drift=0.004), strategy_fn=_straddle, config=cfg)
    assert out.delta_pnl.sum() + out.hedge_pnl.sum() == pytest.approx(0.0, abs=1e-6)


# ── Position lifecycle ───────────────────────────────────────────────────


def test_positions_are_held_across_days():
    """The original engine closed every position daily; it must not."""
    cfg = BacktestConfig(holding_days=5)
    out = run_backtest(_history(20), strategy_fn=_straddle, config=cfg)
    assert (out.n_positions > 0).all()


def test_holding_period_controls_turnover():
    hist = _history(40)
    short_hold = run_backtest(hist, strategy_fn=_straddle,
                              config=BacktestConfig(holding_days=2))
    long_hold = run_backtest(hist, strategy_fn=_straddle,
                             config=BacktestConfig(holding_days=20))
    assert short_hold.transaction_costs.sum() > long_hold.transaction_costs.sum()


def test_costs_scale_with_the_cost_rate():
    hist = _history(30)
    cheap = run_backtest(hist, strategy_fn=_straddle,
                         config=BacktestConfig(transaction_cost_bps=1, slippage_bps=0))
    dear = run_backtest(hist, strategy_fn=_straddle,
                        config=BacktestConfig(transaction_cost_bps=50, slippage_bps=10))
    assert dear.transaction_costs.sum() > cheap.transaction_costs.sum()


def test_zero_costs_when_rates_are_zero():
    out = run_backtest(_history(20), strategy_fn=_straddle,
                       config=BacktestConfig(transaction_cost_bps=0, slippage_bps=0))
    assert out.transaction_costs.sum() == pytest.approx(0.0)


# ── Economics ────────────────────────────────────────────────────────────


def test_long_gamma_profits_from_a_large_move():
    """A delta-hedged long straddle must gain gamma P&L when spot jumps."""
    calm = run_backtest(_history(30, drift=0.0), strategy_fn=_straddle)
    jumpy = run_backtest(_history(30, drift=0.0, shock=1.10), strategy_fn=_straddle)
    assert jumpy.gamma_pnl.sum() > calm.gamma_pnl.sum()


def test_long_straddle_pays_theta():
    out = run_backtest(_history(30), strategy_fn=_straddle)
    assert out.theta_pnl.sum() < 0


def test_short_straddle_earns_theta():
    def short(dt, surface, ticker="TEST", config=None):
        return _straddle(dt, surface, qty=-1.0)

    out = run_backtest(_history(30), strategy_fn=short)
    assert out.theta_pnl.sum() > 0


def test_rising_vol_helps_a_long_vega_position():
    def rising(n=30):
        out, d = {}, date(2026, 1, 5)
        for i in range(n):
            out[d] = _flat_surface(d, spot=100.0, iv=0.20 + 0.002 * i)
            d += timedelta(days=1)
        return out

    out = run_backtest(rising(), strategy_fn=_straddle)
    assert out.vega_pnl.sum() > 0


def test_results_are_finite():
    out = run_backtest(_history(40, drift=0.002, shock=0.92), strategy_fn=_straddle)
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()
