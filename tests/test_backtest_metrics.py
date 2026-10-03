"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    compute_all_metrics,
    log_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _frame(pv, start=1_000_000.0):
    """Build a backtest-shaped frame from a portfolio-value path."""
    pv = np.asarray(pv, dtype=float)
    pnl = np.diff(np.concatenate([[start], pv]))
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(pv), freq="D"),
        "net_pnl": pnl,
        "portfolio_value": pv,
        "gross_pnl": pnl,
        "transaction_costs": np.zeros(len(pv)),
    })


# ── Building blocks ──────────────────────────────────────────────────────


def test_log_returns_length():
    assert len(log_returns([100, 101, 102])) == 2


def test_log_returns_sum_to_total_log_growth():
    pv = [100.0, 120.0, 90.0, 110.0]
    assert log_returns(pv).sum() == pytest.approx(np.log(110.0 / 100.0))


def test_max_drawdown_simple():
    assert max_drawdown(np.array([100.0, 120.0, 60.0, 80.0])) == pytest.approx(0.5)


def test_max_drawdown_monotone_path_is_zero():
    assert max_drawdown(np.array([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_win_rate():
    assert win_rate(np.array([1.0, -1.0, 2.0, 0.0])) == pytest.approx(0.5)


def test_sharpe_of_constant_returns_is_zero():
    assert sharpe_ratio(np.zeros(50)) == 0.0


def test_sharpe_sign_follows_mean_return():
    rng = np.random.default_rng(0)
    assert sharpe_ratio(rng.normal(0.001, 0.01, 500)) > 0
    assert sharpe_ratio(rng.normal(-0.001, 0.01, 500)) < 0


def test_sortino_ignores_upside_volatility():
    """Sortino should exceed Sharpe when the downside is the tamer side."""
    r = np.array([0.05, 0.06, -0.01, 0.07, -0.01] * 20)
    assert sortino_ratio(r) > sharpe_ratio(r)


# ── The regression this fixes ────────────────────────────────────────────


def test_losing_strategy_cannot_post_a_positive_sharpe():
    """A volatile path that ends lower must report a negative Sharpe.

    With simple returns the arithmetic mean exceeds the geometric mean, so
    this path used to show a *positive* Sharpe while losing money.
    """
    pv, v = [], 1_000_000.0
    for i in range(200):
        v *= 1.10 if i % 2 == 0 else 0.905      # ends well below the start
        pv.append(v)

    m = compute_all_metrics(_frame(pv))
    assert m["total_return"] < 0
    assert m["total_pnl"] < 0
    assert m["sharpe"] < 0


def test_sharpe_and_total_return_agree_in_sign():
    rng = np.random.default_rng(3)
    for drift in (0.0015, -0.0015):
        pv = 1_000_000 * np.exp(np.cumsum(rng.normal(drift, 0.02, 300)))
        m = compute_all_metrics(_frame(pv))
        assert np.sign(m["sharpe"]) == np.sign(m["total_return"])


# ── Aggregate metrics ────────────────────────────────────────────────────


def test_total_return_matches_the_path():
    m = compute_all_metrics(_frame([1_100_000.0], start=1_000_000.0))
    assert m["total_return"] == pytest.approx(0.10)


def test_total_pnl_matches_net_pnl_sum():
    df = _frame([1_010_000.0, 1_020_000.0, 1_005_000.0])
    assert compute_all_metrics(df)["total_pnl"] == pytest.approx(df.net_pnl.sum())


def test_cagr_of_a_flat_year_is_zero():
    pv = np.full(252, 1_000_000.0)
    assert compute_all_metrics(_frame(pv))["cagr"] == pytest.approx(0.0, abs=1e-9)


def test_cagr_positive_for_a_growing_path():
    pv = 1_000_000 * np.exp(np.linspace(0, 0.20, 252))
    assert compute_all_metrics(_frame(pv))["cagr"] > 0


def test_all_expected_keys_present():
    m = compute_all_metrics(_frame([1_000_100.0, 1_000_200.0]))
    for key in ("total_pnl", "total_return", "cagr", "sharpe", "sortino",
                "max_drawdown", "win_rate", "calmar", "n_days"):
        assert key in m


def test_metrics_are_finite():
    rng = np.random.default_rng(7)
    pv = 1_000_000 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))
    m = compute_all_metrics(_frame(pv))
    assert all(np.isfinite(v) for v in m.values())


def test_single_day_does_not_blow_up():
    m = compute_all_metrics(_frame([1_000_000.0]))
    assert np.isfinite(m["total_pnl"])
    assert m["sharpe"] == 0.0
