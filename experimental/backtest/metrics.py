"""Performance metrics: Sharpe, Sortino, max drawdown, win rate, P&L attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualise: bool = True) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / 252
    if len(excess) < 2 or np.std(excess) < 1e-12:
        return 0.0
    sr = np.mean(excess) / np.std(excess, ddof=1)
    return float(sr * np.sqrt(252) if annualise else sr)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, annualise: bool = True) -> float:
    """Annualised Sortino ratio using downside deviation."""
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return 0.0
    sr = np.mean(excess) / downside_std
    return float(sr * np.sqrt(252) if annualise else sr)


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction)."""
    if len(portfolio_values) < 2:
        return 0.0
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / np.maximum(peak, 1e-12)
    return float(drawdown.max())


def win_rate(pnl_series: np.ndarray) -> float:
    """Fraction of days with positive P&L."""
    if len(pnl_series) == 0:
        return 0.0
    return float(np.sum(pnl_series > 0) / len(pnl_series))


def calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    if len(returns) == 0:
        return 0.0
    ann_ret = float(np.mean(returns) * 252)
    mdd = max_drawdown(portfolio_values)
    if mdd < 1e-12:
        return 0.0
    return ann_ret / mdd


def pnl_attribution(
    pnl_df: pd.DataFrame,
    strategy_col: str = "strategy",
    pnl_col: str = "net_pnl",
) -> pd.Series:
    """Summarise P&L grouped by strategy."""
    if strategy_col not in pnl_df.columns:
        return pd.Series({"total": pnl_df[pnl_col].sum()})
    return pnl_df.groupby(strategy_col)[pnl_col].sum()


def log_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """Daily log returns of a portfolio-value path.

    Log returns are used for the risk ratios because they aggregate the way
    the portfolio actually compounds.  With simple returns the arithmetic mean
    exceeds the geometric mean whenever the path is volatile, so a strategy
    that ends *below* where it started can still post a positive Sharpe — which
    a delta-hedged straddle with a 45% drawdown did.
    """
    pv = np.asarray(portfolio_values, dtype=float)
    pv = np.maximum(pv, 1e-9)
    return np.diff(np.log(pv))


def compute_all_metrics(backtest_df: pd.DataFrame) -> dict[str, float]:
    """Compute all key metrics from a backtest result DataFrame."""
    pnl = backtest_df["net_pnl"].to_numpy()
    pv = backtest_df["portfolio_value"].to_numpy()

    returns = log_returns(pv)
    n = len(pv)
    start = float(pv[0] - pnl[0]) if n else 0.0   # capital before day one's P&L
    end = float(pv[-1]) if n else 0.0
    total_return = (end / start - 1.0) if start > 0 else 0.0

    years = max(n / 252.0, 1e-9)
    cagr = ((end / start) ** (1 / years) - 1.0) if start > 0 and end > 0 else 0.0

    return {
        "total_pnl": float(pnl.sum()),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(pv),
        "win_rate": win_rate(pnl),
        "calmar": calmar_ratio(returns, pv),
        "n_days": len(pnl),
        "avg_daily_pnl": float(pnl.mean()),
        "vol_daily_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
    }
