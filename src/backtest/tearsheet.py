"""Plotly HTML performance tearsheet generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False

from src.backtest.metrics import compute_all_metrics


def generate_tearsheet(
    backtest_df: pd.DataFrame,
    output_path: str | Path = "tearsheet.html",
    title: str = "Strategy Tearsheet",
) -> Path:
    """Generate an HTML performance tearsheet using Plotly.

    Parameters
    ----------
    backtest_df : DataFrame with columns: date, net_pnl, portfolio_value, gross_pnl
    output_path : output HTML file path
    title       : page title
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = compute_all_metrics(backtest_df)
    pv = backtest_df["portfolio_value"].to_numpy()
    pnl = backtest_df["net_pnl"].to_numpy()
    dates = backtest_df["date"].tolist()

    # Drawdown series
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / np.maximum(peak, 1.0) * 100

    # Rolling Sharpe (63-day)
    returns = pd.Series(pnl / np.maximum(np.roll(pv, 1), 1.0))
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std(ddof=1) * np.sqrt(252)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Cumulative P&L", "Daily P&L", "Drawdown (%)", "Rolling 63-day Sharpe"
        ],
        vertical_spacing=0.08,
    )

    # Cumulative P&L
    fig.add_trace(go.Scatter(x=dates, y=backtest_df["cumulative_pnl"],
                             name="Cumulative P&L", line=dict(color="royalblue")), row=1, col=1)

    # Daily P&L
    colours = ["green" if v >= 0 else "red" for v in pnl]
    fig.add_trace(go.Bar(x=dates, y=pnl, name="Daily P&L",
                         marker_color=colours, showlegend=True), row=2, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(x=dates, y=-drawdown, name="Drawdown",
                             fill="tozeroy", line=dict(color="orange")), row=3, col=1)

    # Rolling Sharpe
    fig.add_trace(go.Scatter(x=dates, y=rolling_sharpe,
                             name="Rolling Sharpe", line=dict(color="purple")), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

    # Summary annotation
    summary = (
        f"Sharpe: {metrics['sharpe']:.2f} | "
        f"Sortino: {metrics['sortino']:.2f} | "
        f"Max DD: {metrics['max_drawdown']:.1%} | "
        f"Win Rate: {metrics['win_rate']:.1%} | "
        f"Total P&L: ${metrics['total_pnl']:,.0f}"
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>{summary}</sup>", x=0.5),
        height=1200,
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(str(output_path))
    return output_path
