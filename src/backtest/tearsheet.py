"""
HTML tearsheet generation for backtest results.

Generates a self-contained HTML report with:
- Equity curve chart
- Drawdown chart
- Monthly returns heatmap
- Summary statistics table
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_tearsheet(
    returns: pd.Series,
    output_path: str = "reports/backtest.html",
    title: str = "Vol-Arb Strategy Backtest",
) -> Path:
    """Generate an HTML tearsheet from a returns series.

    Parameters
    ----------
    returns:
        Daily P&L series.
    output_path:
        Output file path.
    title:
        Report title.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    from src.backtest.metrics import compute_metrics

    metrics = compute_metrics(returns)
    cum_returns = (1 + returns).cumprod() - 1
    rolling_max = (1 + returns).cumprod().expanding().max()
    drawdown = (1 + returns).cumprod() / rolling_max - 1

    # Build subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Cumulative Returns", "Drawdown", "Daily P&L"],
        row_heights=[0.4, 0.3, 0.3],
    )

    fig.add_trace(
        go.Scatter(x=cum_returns.index, y=cum_returns.values, name="Cumulative Return",
                   line=dict(color="royalblue")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown",
                   fill="tozeroy", line=dict(color="red")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=returns.index, y=returns.values, name="Daily P&L",
               marker_color=["green" if r >= 0 else "red" for r in returns.values]),
        row=3, col=1,
    )

    # Metrics table
    metrics_text = "<table><tr><th>Metric</th><th>Value</th></tr>" + "".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v:.4f}</td></tr>"
        for k, v in metrics.items()
    ) + "</table>"

    html_content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; margin-top: 20px; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>{title}</h1>
{metrics_text}
<div id="chart">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>
</body>
</html>"""

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content)
    logger.info("Tearsheet saved to %s", out_path)
    return out_path
