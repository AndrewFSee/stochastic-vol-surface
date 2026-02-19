"""
Performance metrics for backtesting: Sharpe, Sortino, max drawdown, win rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free_rate / annualization
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annualization))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / annualization
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(annualization))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    drawdowns = cum / rolling_max - 1
    return float(drawdowns.min())


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""
    return float((returns > 0).mean())


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Compute a full set of performance metrics.

    Parameters
    ----------
    returns:
        Daily P&L series.
    risk_free_rate:
        Annual risk-free rate.

    Returns
    -------
    dict[str, float]
    """
    return {
        "total_return": float((1 + returns).prod() - 1),
        "annualised_return": float(returns.mean() * 252),
        "annualised_vol": float(returns.std() * np.sqrt(252)),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "n_trades": int(len(returns)),
    }
